# -*- coding: utf-8 -*-
"""
 Created on 2021/7/9 23:51
 Filename   : ex_pcr_vs_pls.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description:
"""

# =======================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression


def plot_pcr_vs_pls():
    rng = np.random.RandomState(0)
    n_samples = 500
    cov = [[3, 3],
           [3, 4]]
    X = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n_samples)
    pca = PCA(n_components=2).fit(X)

    plt.scatter(X[:, 0], X[:, 1], alpha=.3, label='samples')
    for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
        comp = comp * var  # scale component by its variance explanation power
        plt.plot([0, comp[0]], [0, comp[1]], label=f"Component {i}", linewidth=5,
                 color=f"C{i + 2}")
    plt.gca().set(aspect='equal',
                  title="2-dimensional dataset with principal components",
                  xlabel='first feature', ylabel='second feature')
    plt.legend()
    plt.show()

    y = X.dot(pca.components_[1]) + rng.normal(size=n_samples) / 2

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

    axes[0].scatter(X.dot(pca.components_[0]), y, alpha=.3)
    axes[0].set(xlabel='Projected data onto first PCA component', ylabel='y')
    axes[1].scatter(X.dot(pca.components_[1]), y, alpha=.3)
    axes[1].set(xlabel='Projected data onto second PCA component', ylabel='y')
    plt.tight_layout()
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)

    pcr = make_pipeline(StandardScaler(), PCA(n_components=1), LinearRegression())
    pcr.fit(X_train, y_train)
    pca = pcr.named_steps['pca']  # retrieve the PCA step of the pipeline

    pls = PLSRegression(n_components=1)
    pls.fit(X_train, y_train)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].scatter(pca.transform(X_test), y_test, alpha=.3, label='ground truth')
    axes[0].scatter(pca.transform(X_test), pcr.predict(X_test), alpha=.3,
                    label='predictions')
    axes[0].set(xlabel='Projected data onto first PCA component',
                ylabel='y', title='PCR / PCA')
    axes[0].legend()
    axes[1].scatter(pls.transform(X_test), y_test, alpha=.3, label='ground truth')
    axes[1].scatter(pls.transform(X_test), pls.predict(X_test), alpha=.3,
                    label='predictions')
    axes[1].set(xlabel='Projected data onto first PLS component',
                ylabel='y', title='PLS')
    axes[1].legend()
    plt.tight_layout()
    plt.show()

    print(f"PCR r-squared {pcr.score(X_test, y_test):.3f}")
    print(f"PLS r-squared {pls.score(X_test, y_test):.3f}")

    pca_2 = make_pipeline(PCA(n_components=2), LinearRegression())
    pca_2.fit(X_train, y_train)
    print(f"PCR r-squared with 2 components {pca_2.score(X_test, y_test):.3f}")


if __name__ == '__main__':
    plot_pcr_vs_pls()
