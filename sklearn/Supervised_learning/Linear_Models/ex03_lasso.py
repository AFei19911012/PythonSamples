# -*- coding: utf-8 -*-
"""
 Created on 2021/7/4 22:47
 Filename   : ex03_lasso.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description:
"""

# =======================================================
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from scipy import sparse, ndimage

""" min{1/(2*n)*norm(Xw - y, 2)^2 + α*norm(w, 1)} """


def plot_lasso():
    reg = linear_model.Lasso(alpha=0.1)
    reg.fit([[0, 0], [1, 1]], [0, 1])
    print(reg.predict([[1, 1]]))


def plot_lasso_and_elasticnet():
    np.random.seed(42)
    n_samples, n_features = 50, 100
    X = np.random.randn(n_samples, n_features)

    # Decreasing coef w. alternated signs for visualization
    idx = np.arange(n_features)
    coef = (-1) ** idx * np.exp(-idx / 10)
    coef[10:] = 0  # sparsify coef
    y = np.dot(X, coef)

    # Add noise
    y += 0.01 * np.random.normal(size=n_samples)

    # Split data in train set and test set
    n_samples = X.shape[0]
    X_train, y_train = X[:n_samples // 2], y[:n_samples // 2]
    X_test, y_test = X[n_samples // 2:], y[n_samples // 2:]

    # Lasso
    alpha = 0.1
    lasso = Lasso(alpha=alpha)

    y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
    r2_score_lasso = r2_score(y_test, y_pred_lasso)
    print(lasso)
    print(f"r^2 on test data : {r2_score_lasso}")

    # ElasticNet
    from sklearn.linear_model import ElasticNet

    enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

    y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
    r2_score_enet = r2_score(y_test, y_pred_enet)
    print(enet)
    print(f"r^2 on test data : {r2_score_enet}")

    m, s, _ = plt.stem(np.where(enet.coef_)[0], enet.coef_[enet.coef_ != 0], markerfmt='x', label='Elastic net coefficients',
                       use_line_collection=True)
    plt.setp([m, s], color="#2ca02c")
    m, s, _ = plt.stem(np.where(lasso.coef_)[0], lasso.coef_[lasso.coef_ != 0], markerfmt='x', label='Lasso coefficients',
                       use_line_collection=True)
    plt.setp([m, s], color='#ff7f0e')
    plt.stem(np.where(coef)[0], coef[coef != 0], label='true coefficients', markerfmt='bx', use_line_collection=True)

    plt.legend(loc='best')
    plt.title(f"Lasso $R^2$: {r2_score_lasso:.3f}, Elastic Net $R^2$: {r2_score_enet:.3f}")
    plt.show()


def plot_tomography_l1_reconstruction():
    def _weights(x, dx=1, orig=0):
        x = np.ravel(x)
        floor_x = np.floor((x - orig) / dx).astype(np.int64)
        alpha = (x - orig - floor_x * dx) / dx
        return np.hstack((floor_x, floor_x + 1)), np.hstack((1 - alpha, alpha))

    def _generate_center_coordinates(l_x):
        X, Y = np.mgrid[:l_x, :l_x].astype(np.float64)
        center = l_x / 2.
        X += 0.5 - center
        Y += 0.5 - center
        return X, Y

    def build_projection_operator(l_x, n_dir):
        """ Compute the tomography design matrix.

        Parameters
        ----------

        l_x : int
            linear size of image array

        n_dir : int
            number of angles at which projections are acquired.

        Returns
        -------
        p : sparse matrix of shape (n_dir l_x, l_x**2)
        """
        X, Y = _generate_center_coordinates(l_x)
        angles = np.linspace(0, np.pi, n_dir, endpoint=False)
        data_inds, weights, camera_inds = [], [], []
        data_unravel_indices = np.arange(l_x ** 2)
        data_unravel_indices = np.hstack((data_unravel_indices,
                                          data_unravel_indices))
        for i, angle in enumerate(angles):
            Xrot = np.cos(angle) * X - np.sin(angle) * Y
            inds, w = _weights(Xrot, dx=1, orig=X.min())
            mask = np.logical_and(inds >= 0, inds < l_x)
            weights += list(w[mask])
            camera_inds += list(inds[mask] + i * l_x)
            data_inds += list(data_unravel_indices[mask])
        proj_operator = sparse.coo_matrix((weights, (camera_inds, data_inds)))
        return proj_operator

    def generate_synthetic_data():
        """ Synthetic binary data """
        rs = np.random.RandomState(0)
        n_pts = 36
        x, y = np.ogrid[0:l, 0:l]
        mask_outer = (x - l / 2.) ** 2 + (y - l / 2.) ** 2 < (l / 2.) ** 2
        mask = np.zeros((l, l))
        points = l * rs.rand(2, n_pts)
        mask[(points[0]).astype(int), (points[1]).astype(int)] = 1
        mask = ndimage.gaussian_filter(mask, sigma=l / n_pts)
        res = np.logical_and(mask > mask.mean(), mask_outer)
        return np.logical_xor(res, ndimage.binary_erosion(res))

    # Generate synthetic images, and projections
    l = 128
    proj_operator = build_projection_operator(l, l // 7)
    data = generate_synthetic_data()
    proj = proj_operator @ data.ravel()[:, np.newaxis]
    proj += 0.15 * np.random.randn(*proj.shape)

    # Reconstruction with L2 (Ridge) penalization
    rgr_ridge = Ridge(alpha=0.2)
    rgr_ridge.fit(proj_operator, proj.ravel())
    rec_l2 = rgr_ridge.coef_.reshape(l, l)

    # Reconstruction with L1 (Lasso) penalization
    # the best value of alpha was determined using cross validation
    # with LassoCV
    rgr_lasso = Lasso(alpha=0.001)
    rgr_lasso.fit(proj_operator, proj.ravel())
    rec_l1 = rgr_lasso.coef_.reshape(l, l)

    plt.figure(figsize=(8, 3.3))
    plt.subplot(131)
    plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
    plt.axis('off')
    plt.title('original image')
    plt.subplot(132)
    plt.imshow(rec_l2, cmap=plt.cm.gray, interpolation='nearest')
    plt.title('L2 penalization')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(rec_l1, cmap=plt.cm.gray, interpolation='nearest')
    plt.title('L1 penalization')
    plt.axis('off')

    plt.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)

    plt.show()


def plot_lasso_model_selection():
    """ https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html#sphx-glr-auto-examples-linear-model-plot-lasso
    -model-selection-py """
    pass


if __name__ == '__main__':
    # plot_lasso()
    # plot_lasso_and_elasticnet()
    plot_tomography_l1_reconstruction()
