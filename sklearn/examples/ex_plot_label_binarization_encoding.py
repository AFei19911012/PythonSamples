# -*- coding: utf-8 -*-
"""
 Created on 2021/7/10 14:48
 Filename   : ex_plot_label_binarization_encoding.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description:
"""

# =======================================================
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer


def plot_label_binarization_encoding():
    lb = preprocessing.LabelBinarizer()
    lb.fit([1, 2, 6, 4, 2])

    print(lb.classes_)
    print(lb.transform([1, 6]))

    y = [[2, 3, 4], [2], [0, 1, 3], [0, 1, 2, 3, 4], [0, 1, 2]]
    print(MultiLabelBinarizer().fit_transform(y))

    le = preprocessing.LabelEncoder()
    le.fit([1, 2, 2, 6])

    print(le.classes_)
    print(le.transform([1, 1, 2, 6]))
    print(le.inverse_transform([0, 0, 1, 2]))

    le = preprocessing.LabelEncoder()
    le.fit(["paris", "paris", "tokyo", "amsterdam"])

    print(list(le.classes_))

    le.transform(["tokyo", "tokyo", "paris"])

    list(le.inverse_transform([2, 2, 1]))
