# -*- coding: utf-8 -*-
"""
 Created on 2021/7/8 23:46
 Filename   : ex01_sgd_classifier.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description:
"""

# =======================================================
from sklearn.linear_model import SGDClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm


def plot_sgd_classifier():
    X = [[0., 0.], [1., 1.]]
    y = [0, 1]
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    clf.fit(X, y)

    print(clf.predict([[2., 2.]]))
    print(clf.coef_)
    print(clf.intercept_)

    clf = SGDClassifier(loss="log", max_iter=5).fit(X, y)
    print(clf.predict_proba([[1., 1.]]))


def plot_sgd_separating_hyperplane():
    # we create 50 separable points
    X, Y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)

    # fit the model
    clf = SGDClassifier(loss="hinge", alpha=0.01, max_iter=200)

    clf.fit(X, Y)

    # plot the line, the points, and the nearest vectors to the plane
    xx = np.linspace(-1, 5, 10)
    yy = np.linspace(-1, 5, 10)

    X1, X2 = np.meshgrid(xx, yy)
    Z = np.empty(X1.shape)
    for (i, j), val in np.ndenumerate(X1):
        x1 = val
        x2 = X2[i, j]
        p = clf.decision_function([[x1, x2]])
        Z[i, j] = p[0]
    levels = [-1.0, 0.0, 1.0]
    linestyles = ['dashed', 'solid', 'dashed']
    colors = 'k'
    plt.contour(X1, X2, Z, levels, colors=colors, linestyles=linestyles)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired,
                edgecolor='black', s=20)

    plt.axis('tight')
    plt.show()


def plot_sgd_iris():
    # import some data to play with
    iris = datasets.load_iris()

    # we only take the first two features. We could
    # avoid this ugly slicing by using a two-dim dataset
    X = iris.data[:, :2]
    y = iris.target
    colors = "bry"

    # shuffle
    idx = np.arange(X.shape[0])
    np.random.seed(13)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    # standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    h = .02  # step size in the mesh

    clf = SGDClassifier(alpha=0.001, max_iter=100).fit(X, y)

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis('tight')

    # Plot also the training points
    for i, color in zip(clf.classes_, colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i], cmap=plt.cm.Paired, edgecolor='black', s=20)
    plt.title("Decision surface of multi-class SGD")
    plt.axis('tight')

    # Plot the three one-against-all classifiers
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    coef = clf.coef_
    intercept = clf.intercept_

    def plot_hyperplane(c, color):
        def line(x0):
            return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]

        plt.plot([xmin, xmax], [line(xmin), line(xmax)], ls="--", color=color)

    for i, color in zip(clf.classes_, colors):
        plot_hyperplane(i, color)
    plt.legend()
    plt.show()


def plot_sgd_weighted_samples():
    # we create 20 points
    np.random.seed(0)
    X = np.r_[np.random.randn(10, 2) + [1, 1], np.random.randn(10, 2)]
    y = [1] * 10 + [-1] * 10
    sample_weight = 100 * np.abs(np.random.randn(20))
    # and assign a bigger weight to the last 10 samples
    sample_weight[:10] *= 10

    # plot the weighted data points
    xx, yy = np.meshgrid(np.linspace(-4, 5, 500), np.linspace(-4, 5, 500))
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=sample_weight, alpha=0.9, cmap=plt.cm.bone, edgecolor='black')

    # fit the unweighted model
    clf = linear_model.SGDClassifier(alpha=0.01, max_iter=100)
    clf.fit(X, y)
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    no_weights = plt.contour(xx, yy, Z, levels=[0], linestyles=['solid'])

    # fit the weighted model
    clf = linear_model.SGDClassifier(alpha=0.01, max_iter=100)
    clf.fit(X, y, sample_weight=sample_weight)
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    samples_weights = plt.contour(xx, yy, Z, levels=[0], linestyles=['dashed'])

    plt.legend([no_weights.collections[0], samples_weights.collections[0]], ["no weights", "with weights"], loc="lower left")

    plt.xticks(())
    plt.yticks(())
    plt.show()


def plot_sgd_comparison():
    heldout = [0.95, 0.90, 0.75, 0.50, 0.01]
    rounds = 20
    X, y = datasets.load_digits(return_X_y=True)

    classifiers = [
        ("SGD", SGDClassifier(max_iter=100)),
        ("ASGD", SGDClassifier(average=True)),
        ("Perceptron", Perceptron()),
        ("Passive-Aggressive I", PassiveAggressiveClassifier(loss='hinge',
                                                             C=1.0, tol=1e-4)),
        ("Passive-Aggressive II", PassiveAggressiveClassifier(loss='squared_hinge',
                                                              C=1.0, tol=1e-4)),
        ("SAG", LogisticRegression(solver='sag', tol=1e-1, C=1.e4 / X.shape[0]))
    ]

    xx = 1. - np.array(heldout)

    for name, clf in classifiers:
        print("training %s" % name)
        rng = np.random.RandomState(42)
        yy = []
        for i in heldout:
            yy_ = []
            for r in range(rounds):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i, random_state=rng)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                yy_.append(1 - np.mean(y_pred == y_test))
            yy.append(np.mean(yy_))
        plt.plot(xx, yy, label=name)

    plt.legend(loc="upper right")
    plt.xlabel("Proportion train")
    plt.ylabel("Test Error Rate")
    plt.show()


def plot_separating_hyperplane_unbalanced():
    # we create two clusters of random points
    n_samples_1 = 1000
    n_samples_2 = 100
    centers = [[0.0, 0.0], [2.0, 2.0]]
    clusters_std = [1.5, 0.5]
    X, y = make_blobs(n_samples=[n_samples_1, n_samples_2], centers=centers, cluster_std=clusters_std, random_state=0, shuffle=False)

    # fit the model and get the separating hyperplane
    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(X, y)

    # fit the model and get the separating hyperplane using weighted classes
    wclf = svm.SVC(kernel='linear', class_weight={1: 10})
    wclf.fit(X, y)

    # plot the samples
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')

    # plot the decision functions for both classifiers
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    # get the separating hyperplane
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    a = ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

    # get the separating hyperplane for weighted classes
    Z = wclf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins for weighted classes
    b = ax.contour(XX, YY, Z, colors='r', levels=[0], alpha=0.5, linestyles=['-'])

    plt.legend([a.collections[0], b.collections[0]], ["non weighted", "weighted"], loc="upper right")
    plt.show()


if __name__ == '__main__':
    plot_sgd_separating_hyperplane()
    plot_sgd_weighted_samples()
    plot_separating_hyperplane_unbalanced()
