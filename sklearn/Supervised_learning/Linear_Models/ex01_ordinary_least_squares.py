# -*- coding: utf-8 -*-
"""
 Created on 2021/7/4 17:34
 Filename   : ex01_ordinary_least_squares.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description: 最小二乘法
"""

# =======================================================
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

""" y = w0 + w1*x1 + w2*x2 + ... + wn*xn """
""" min{norm(Xw - y, 2)^2} """


def linear_regression():
    """ 线性回归 """
    reg = linear_model.LinearRegression()
    reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
    print(reg.coef_)


def ordinary_least_squares():
    """ OLS 最小二乘法 """

    # Load the diabetes dataset
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

    # Use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)

    # The coefficients
    print(f'Coefficients: {regr.coef_}')
    # The mean squared error
    print(f'Mean squared error: {mean_squared_error(diabetes_y_test, diabetes_y_pred):.2f}')
    # The coefficient of determination: 1 is perfect prediction
    print(f'Coefficient of determination: {r2_score(diabetes_y_test, diabetes_y_pred):.2f}')

    # Plot outputs
    plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
    plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()


def non_negative_least_squares():
    """ 非负最小二乘 """
    np.random.seed(42)
    n_samples, n_features = 200, 50
    X = np.random.randn(n_samples, n_features)
    true_coef = 3 * np.random.randn(n_features)

    # Threshold coefficients to render them non-negative
    true_coef[true_coef < 0] = 0
    y = np.dot(X, true_coef)

    # Add some noise
    y += 5 * np.random.normal(size=(n_samples,))

    # Split the targets into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    """ Fit the Non-Negative least squares """
    reg_nnls = linear_model.LinearRegression(positive=True)
    y_pred_nnls = reg_nnls.fit(X_train, y_train).predict(X_test)
    r2_score_nnls = r2_score(y_test, y_pred_nnls)
    print(f"NNLS R2 score: {r2_score_nnls}")

    """ Fit an OLS """
    reg_ols = linear_model.LinearRegression()
    y_pred_ols = reg_ols.fit(X_train, y_train).predict(X_test)
    r2_score_ols = r2_score(y_test, y_pred_ols)
    print(f"OLS R2 score: {r2_score_ols}")

    """ Comparing the regression coefficients between OLS and NNLS """
    fig, ax = plt.subplots()
    ax.plot(reg_ols.coef_, reg_nnls.coef_, linewidth=0, marker=".")
    low_x, high_x = ax.get_xlim()
    low_y, high_y = ax.get_ylim()
    low = max(low_x, low_y)
    high = min(high_x, high_y)
    ax.plot([low, high], [low, high], ls="--", c=".3", alpha=.5)
    ax.set_xlabel("OLS regression coefficients", fontweight="bold")
    ax.set_ylabel("NNLS regression coefficients", fontweight="bold")
    plt.show()


if __name__ == '__main__':
    # linear_regression()
    # ordinary_least_squares()
    non_negative_least_squares()
