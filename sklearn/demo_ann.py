# -*- coding: utf-8 -*-
"""
 Created on 2021/6/30 21:24
 Filename   : demo_ann.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description: 乳腺癌数据集的人工神经网络
"""

import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def breast_cancer_ann():
    """ 乳腺癌数据集的人工神经网络 """
    """ 读取数据 """
    df = pd.read_csv('dataset/breast_cancer.csv')

    """ 删除空值 """
    print(df.isnull().sum())
    df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)

    """ 自变量和因变量 """
    x = df.drop('diagnosis', axis=1)
    y = df.diagnosis

    """ 分类数据转换为二进制格式 """
    lb = LabelEncoder()
    y = lb.fit_transform(y)

    """ 拆分数据成训练集和测试集 """
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=40)

    """ 数据标准化 """
    sc = StandardScaler()
    xtrain = sc.fit_transform(xtrain)
    xtest = sc.transform(xtest)

    """ 创建 ann """
    """ 创建模型 Sequential 模型适用于每一层恰好有一个输入张量和一个输出张量的平面堆栈 """
    classifier = Sequential()

    """ 创建神经网络层：两个隐含层、一个输出层 """
    """ 四个参数：输出节点、内核权重矩阵的初始化器、激活函数、输入节点数（独立特征数） """
    classifier.add(Dense(units=9, kernel_initializer='he_uniform', activation='relu', input_dim=30))
    classifier.add(Dense(units=9, kernel_initializer='he_uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))

    """ 总结 """
    classifier.summary()

    """ 编译 """
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    """ 训练数据拟合 """
    model = classifier.fit(xtrain, ytrain, batch_size=100, epochs=100)

    """ 测试集 """
    y_pred = classifier.predict(xtest)

    """ 二分类 """
    y_pred = y_pred > 0.5

    """ 检查混淆矩阵和预测值得分 """
    cm = confusion_matrix(ytest, y_pred)
    score = accuracy_score(ytest, y_pred)
    print(cm)
    print(f'score is: {score}')

    """ 历史数据可视化 """
    print(model.history.keys())
    plt.plot(model.history['accuracy'])
    plt.plot(model.history['loss'])
    plt.title('model accuracy')
    plt.ylabel('value')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'loss'], loc='upper left')
    plt.show()

    """ 保存模型 """
    classifier.save('models/breast_cancer.h5')


if __name__ == '__main__':
    breast_cancer_ann()
