# -*- coding: utf-8 -*-
"""
 Created on 2021/7/1 20:34
 Filename   : demo_mnist_cnn.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description:
"""

# =======================================================
from __future__ import print_function
import os
import random
import keras
import numpy as np
from PIL import Image
from keras.layers.core import Dense, Activation, Flatten
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator


def load_data():
    """ 读取图像数据 """
    """ 读取文件夹 mnist 下的 42000 张图片，图片为灰度图，图像大小 28*28 """
    data = np.empty((42000, 28, 28, 1), dtype="float32")
    label = np.empty((42000,), dtype="uint8")
    imgs = os.listdir("D:/MyPrograms/DataSet/mnist")
    num = len(imgs)
    for i in range(num):
        img = Image.open("D:/MyPrograms/DataSet/mnist/" + imgs[i])
        arr = np.asarray(img, dtype="float32")
        data[i, :, :, 0] = arr
        label[i] = int(imgs[i].split('.')[0])
    data /= np.max(data)
    data -= np.mean(data)
    return data, label


def data_augmentation(x_train):
    datagen = ImageDataGenerator(
        featurewise_center=False,             # 将整个数据集的均值设为0
        samplewise_center=False,              # 将每个样本的均值设为0
        featurewise_std_normalization=False,  # 将输入除以整个数据集的标准差
        samplewise_std_normalization=False,   # 将输入除以其标准差
        zca_whitening=False,                  # 运用 ZCA 白化
        zca_epsilon=1e-06,                    # ZCA 白化的 epsilon 值
        rotation_range=0,                     # 随机旋转图像范围 (角度, 0 to 180)
        width_shift_range=0.1,                # 随机水平移动图像 (总宽度的百分比)
        height_shift_range=0.1,               # 随机垂直移动图像 (总高度的百分比)
        shear_range=0.,                       # 设置随机裁剪范围
        zoom_range=0.,                        # 设置随机放大范围
        channel_shift_range=0.,               # 设置随机通道切换的范围
        fill_mode='nearest',                  # 设置填充输入边界之外的点的模式
        cval=0.,                              # 在 fill_mode = "constant" 时使用的值
        horizontal_flip=True,                 # 随机水平翻转图像
        vertical_flip=False,                  # 随机垂直翻转图像
        rescale=None,                         # 设置缩放因子 (在其他转换之前使用)
        preprocessing_function=None,          # 设置将应用于每一个输入的函数
        data_format=None,                     # 图像数据格式，"channels_first" 或 "channels_last" 之一
        validation_split=0.0)                 # 保留用于验证的图像比例（严格在 0 和 1 之间）
    datagen.fit(x_train)
    return datagen


def train():
    """ 训练模型 """
    """ 加载数据 """
    data, label = load_data()
    print(data.shape[0], ' samples')

    """ label 为 0~9 共 10 个类别，keras 要求格式为 binary class matrices """
    label = keras.utils.to_categorical(label, 10)
    print(data.shape[1:])
    """ 构建 CNN 模型 """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=data.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    """ 初始化 RMSprop 优化器 """
    opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

    """ 利用 RMSprop 来训练模型 """
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    """ 将 20% 的数据作为验证集 """
    model.fit(data, label, batch_size=32, epochs=100, shuffle=True, verbose=1, validation_split=0.2)
    # model.fit(data, label, batch_size=32, epochs=100, validation_data=(x_test, y_test), shuffle=True)
    """ 数据增强 """
    # datagen = data_augmentation(data)
    # model.fit_generator(datagen.flow(data, label, batch_size=32), epochs=100, validation_split=0.2, workers=4)

    """ 保存模型 """
    model.save('models/mnist_cnn.h5')
    print('Saved trained model at: models/mnist_cnn.h5')

    """ 评估训练模型 """
    scores = model.evaluate(data, label, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


def test():
    """ 模型测试 """
    """ 加载模型 """
    model = load_model('models/mnist_cnn.h5')
    """ 加载测试数据 """
    data, label = load_data()
    """ 随机测试 """
    accu = 0
    x = np.empty((1, 28, 28, 1), dtype="float32")
    for i in range(0, 100):
        index = random.randint(0, data.shape[0])
        x[0] = data[index]
        y = label[index]
        """ 显示数字 """
        # plt.imshow(x)
        # plt.title(f'number {y}')
        # plt.show()
        """ 预处理 """
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
