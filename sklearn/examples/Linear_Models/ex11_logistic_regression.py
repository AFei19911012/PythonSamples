# -*- coding: utf-8 -*-
"""
 Created on 2021/7/5 23:19
 Filename   : ex11_logistic_regression.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description:
"""

# =======================================================
import timeit
import warnings

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
import time

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state


def plot_logistic_l1_l2_sparsity():
    """ https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_l1_l2_sparsity.html#sphx-glr-auto-examples-linear-model-plot
    -logistic-l1-l2-sparsity-py """
    pass


def plot_logistic_path():
    """ https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_path.html#sphx-glr-auto-examples-linear-model-plot-logistic-path
    -py """
    pass


def plot_logistic_multinomial():
    """ https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_multinomial.html#sphx-glr-auto-examples-linear-model-plot-logistic
    -multinomial-py """
    pass


def plot_sparse_logistic_regression_20newsgroups():
    warnings.filterwarnings("ignore", category=ConvergenceWarning,
                            module="sklearn")
    t0 = timeit.default_timer()

    # We use SAGA solver
    solver = 'saga'

    # Turn down for faster run time
    n_samples = 10000

    X, y = fetch_20newsgroups_vectorized(subset='all', return_X_y=True)
    X = X[:n_samples]
    y = y[:n_samples]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=42,
                                                        stratify=y,
                                                        test_size=0.1)
    train_samples, n_features = X_train.shape
    n_classes = np.unique(y).shape[0]

    print('Dataset 20newsgroup, train_samples=%i, n_features=%i, n_classes=%i'
          % (train_samples, n_features, n_classes))

    models = {'ovr': {'name': 'One versus Rest', 'iters': [1, 2, 4]},
              'multinomial': {'name': 'Multinomial', 'iters': [1, 3, 7]}}

    for model in models:
        # Add initial chance-level values for plotting purpose
        accuracies = [1 / n_classes]
        times = [0]
        densities = [1]

        model_params = models[model]

        # Small number of epochs for fast runtime
        for this_max_iter in model_params['iters']:
            print('[model=%s, solver=%s] Number of epochs: %s' %
                  (model_params['name'], solver, this_max_iter))
            lr = LogisticRegression(solver=solver,
                                    multi_class=model,
                                    penalty='l1',
                                    max_iter=this_max_iter,
                                    random_state=42,
                                    )
            t1 = timeit.default_timer()
            lr.fit(X_train, y_train)
            train_time = timeit.default_timer() - t1

            y_pred = lr.predict(X_test)
            accuracy = np.sum(y_pred == y_test) / y_test.shape[0]
            density = np.mean(lr.coef_ != 0, axis=1) * 100
            accuracies.append(accuracy)
            densities.append(density)
            times.append(train_time)
        models[model]['times'] = times
        models[model]['densities'] = densities
        models[model]['accuracies'] = accuracies
        print('Test accuracy for model %s: %.4f' % (model, accuracies[-1]))
        print('%% non-zero coefficients for model %s, '
              'per class:\n %s' % (model, densities[-1]))
        print('Run time (%i epochs) for model %s:'
              '%.2f' % (model_params['iters'][-1], model, times[-1]))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for model in models:
        name = models[model]['name']
        times = models[model]['times']
        accuracies = models[model]['accuracies']
        ax.plot(times, accuracies, marker='o',
                label='Model: %s' % name)
        ax.set_xlabel('Train time (s)')
        ax.set_ylabel('Test accuracy')
    ax.legend()
    fig.suptitle('Multinomial vs One-vs-Rest Logistic L1\n'
                 'Dataset %s' % '20newsgroups')
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    run_time = timeit.default_timer() - t0
    print('Example run in %.3f s' % run_time)
    plt.show()


def plot_sparse_logistic_regression_mnist():
    # Turn down for faster convergence
    t0 = time.time()
    train_samples = 5000

    # Load data from https://www.openml.org/d/554
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

    random_state = check_random_state(0)
    permutation = random_state.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]
    X = X.reshape((X.shape[0], -1))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_samples, test_size=10000)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Turn up tolerance for faster convergence
    clf = LogisticRegression(
        C=50. / train_samples, penalty='l1', solver='saga', tol=0.1
    )
    clf.fit(X_train, y_train)
    sparsity = np.mean(clf.coef_ == 0) * 100
    score = clf.score(X_test, y_test)
    # print('Best C % .4f' % clf.C_)
    print("Sparsity with L1 penalty: %.2f%%" % sparsity)
    print("Test score with L1 penalty: %.4f" % score)

    coef = clf.coef_.copy()
    plt.figure(figsize=(10, 5))
    scale = np.abs(coef).max()
    for i in range(10):
        l1_plot = plt.subplot(2, 5, i + 1)
        l1_plot.imshow(coef[i].reshape(28, 28), interpolation='nearest',
                       cmap=plt.cm.RdBu, vmin=-scale, vmax=scale)
        l1_plot.set_xticks(())
        l1_plot.set_yticks(())
        l1_plot.set_xlabel('Class %i' % i)
    plt.suptitle('Classification vector for...')

    run_time = time.time() - t0
    print('Example run in %.3f s' % run_time)
    plt.show()


if __name__ == '__main__':
    # plot_sparse_logistic_regression_20newsgroups()
    plot_sparse_logistic_regression_mnist()
