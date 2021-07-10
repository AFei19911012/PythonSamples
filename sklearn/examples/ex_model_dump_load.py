# -*- coding: utf-8 -*-
"""
 Created on 2021/7/10 14:57
 Filename   : ex_model_dump_load.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description:
"""

# =======================================================
from sklearn import svm
from sklearn import datasets
import pickle
from joblib import dump, load


def model_dump_load():
    clf = svm.SVC()
    X, y = datasets.load_iris(return_X_y=True)
    clf.fit(X, y)

    s = pickle.dumps(clf)
    clf2 = pickle.loads(s)
    clf2.predict(X[0:1])
    y[0]
    clf = load('filename.joblib')


if __name__ == '__main__':
    model_dump_load()
