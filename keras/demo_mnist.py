# -*- coding: utf-8 -*-
"""
 Created on 2021/6/21 10:41
 Filename   : demo_mnist.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description: 手写体识别
"""

# =======================================================
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras import optimizers
from keras.datasets import mnist
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import random


def load_mnist_dataset(npz_path):
    """ 从本地加载 mnist.npz 数据集 """
    with np.load(npz_path, allow_pickle=True) as f:
        x_train_, y_train_ = f['x_train'], f['y_train']
        x_test_, y_test_ = f['x_test'], f['y_test']
    return (x_train_, y_train_), (x_test_, y_test_)


def train():
    """ 模型训练 """
    """ 自动下载 mnist.npz 数据集 """
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    """ 本地加载 """
    (x_train, y_train), (x_test, y_test) = load_mnist_dataset('dataset/mnist.npz')
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    plt.imshow(x_train[0])
    plt.show()
    """ image --> vector """
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train[0])
    """ 归一化处理 """
    x_train = x_train / 255
    x_test = x_test / 255
    """ 标签处理 5 --> [ 0, 0, 0, 0, 0,1, 0, 0, 0, 0] """
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    """ 构建模型 """
    model = Sequential()
    """ 添加神经网络层 """
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    """ 网络编译 """
    model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
    """ 模型训练 """
    model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_test, y_test))
    """ 模型评价 """
    score = model.evaluate(x_test, y_test)
    """ 保存模型 """
    model.save('models/mnist_model.h5')
    print("loss:", score[0])
    print("accu:", score[1])


def test():
    """ 模型测试 """
    """ 加载模型 """
    model = load_model('models/mnist_model.h5')
    """ 加载测试数据 """
    (_, _), (x_test, y_test) = load_mnist_dataset('dataset/mnist.npz')
    """ 随机测试 """
    accu = 0
    for i in range(0, 100):
        index = random.randint(0, x_test.shape[0])
        x = x_test[index]
        y = y_test[index]
        """ 显示数字 """
        # plt.imshow(x)
        # plt.title(f'number {y}')
        # plt.show()
        """ 预处理 """
        x.shape = (1, 784)
        predict = model.predict(x)
        predict = np.argmax(predict)
        """ 计算正确率 """
        if y == predict:
            accu += 1
        print(f'number: {y}')
        print(f'predicted: {predict}')
        print(' --- ')
    print(f'accu = {accu / 100}')


if __name__ == '__main__':
    # train()
    test()
