# -*- coding: utf-8 -*-
"""
 Created on 2021/7/10 0:06
 Filename   : ex_plot_voting_regressor.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description:
"""

# =======================================================
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor


def plot_voting_regressor():
    X, y = load_diabetes(return_X_y=True)

    # Train classifiers
    reg1 = GradientBoostingRegressor(random_state=1)
    reg2 = RandomForestRegressor(random_state=1)
    reg3 = LinearRegression()

    reg1.fit(X, y)
    reg2.fit(X, y)
    reg3.fit(X, y)

    ereg = VotingRegressor([('gb', reg1), ('rf', reg2), ('lr', reg3)])
    ereg.fit(X, y)

    """ Making predictions """
    xt = X[:20]

    pred1 = reg1.predict(xt)
    pred2 = reg2.predict(xt)
    pred3 = reg3.predict(xt)
    pred4 = ereg.predict(xt)

    """ Plot the results """
    plt.figure()
    plt.plot(pred1, 'gd', label='GradientBoostingRegressor')
    plt.plot(pred2, 'b^', label='RandomForestRegressor')
    plt.plot(pred3, 'ys', label='LinearRegression')
    plt.plot(pred4, 'r*', ms=10, label='VotingRegressor')

    plt.tick_params(axis='x', which='both', bottom=False, top=False,
                    labelbottom=False)
    plt.ylabel('predicted')
    plt.xlabel('training samples')
    plt.legend(loc="best")
    plt.title('Regressor predictions and their average')

    plt.show()