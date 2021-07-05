# -*- coding: utf-8 -*-
"""
 Created on 2021/7/5 23:30
 Filename   : ex12_generalized_linear_regression.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description:
"""

# =======================================================
from sklearn.linear_model import TweedieRegressor


def tweedie_regression():
    reg = TweedieRegressor(power=1, alpha=0.5, link='log')
    reg.fit([[0, 0], [0, 1], [2, 2]], [0, 1, 2])
    print(reg.coef_)
    print(reg.intercept_)


def plot_poisson_regression_non_normal_loss():
    """ https://scikit-learn.org/stable/auto_examples/linear_model/plot_poisson_regression_non_normal_loss.html#sphx-glr-auto-examples-linear-model
    -plot-poisson-regression-non-normal-loss-py """
    pass


def plot_tweedie_regression_insurance_claims():
    """ https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html#sphx-glr-auto-examples-linear
    -model-plot-tweedie-regression-insurance-claims-py """
    pass


if __name__ == '__main__':
    tweedie_regression()
