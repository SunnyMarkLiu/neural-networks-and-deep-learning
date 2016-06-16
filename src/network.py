#!D:\Miniconda2\python
# encoding: utf-8

"""
@author: MarkLiu
@file: network.py
@time: 2016/6/16 20:25
"""
import numpy as np


def sigmoid_activate(z):
    """
    sigmoid激活函数
    :return:
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    """
    sigmoid激活函数的求导
    :return:
    """
    return sigmoid_activate(z) * (1 - sigmoid_activate(z))


class Network(object):
    """
    Simplest Neural Network
    """

    def __init__(self, sizes):
        """
        :type sizes: list
        :param sizes: 神经网络的模型，有多少层，每层有多少神经元，
                    如：Network([10, 4, 1])：该神经网络有三层，输入层（10个神经元），
                    一个隐藏层（4个神经元），一个输出层（1个神经元）
        """
        self.sizes = sizes
        self.layers_num = len(sizes)
        # biases : 隐藏层和输出层的偏置，方便矩阵计算，每层的biase为vector
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, inputs):
        """
        前向传播算法，对于输入input(行向量)，计算最终的输出 a′=σ(wa+b).

        :param inputs:
        :return: 输出层的输出
        """
        a = inputs
        for w, b in zip(self.weights, self.biases):
            a = sigmoid_activate(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, validation_data):
        """
        stochastic gradient descent
        :param training_data: 训练数据，tuple的list集合，每个tuple包括(data, label)
        :param epochs: gradient descent迭代的次数
        :param mini_batch_size: 计算梯度下降的数据集大小
        :param eta: 学习速率
        :param validation_data: 验证集数据
        :return:
        """
        m = len(training_data)
        # 对于每次运行 SGD
        for i in xrange(epochs):
            np.random.shuffle(training_data)  # 将训练数据进行 shuffle!
            # 将数据集按照mini_batch_size进行切分
            mini_batches = [training_data[k:k + mini_batch_size]
                            for k in xrange(0, m, mini_batch_size)]

            for mini_batch in mini_batches:  # 遍历每一份 mini_batch 训练数据
                # 计算损失函数对w,b的偏导数（反相传播算法），更新w,b
                self.update_mini_batch(mini_batch, eta)

            # 训练完毕，计算验证集上的分类错误率
            if validation_data:
                print 'Epoch {0}: {1} / {2}'.format(i, self.classify(validation_data), len(validation_data))
            else:
                print 'Epoch {0} completed'.format(i)

    def update_mini_batch(self, mini_batch, eta):
        pass

    def classify(self, test_data):
        """
        分类
        :param test_data: tuple的list集合，每个tuple包括(data, label)
        :return:
        """
        results = [(np.argmax(self.feedforward(x)), y)
                   for x, y in test_data]
        return sum([int(x == y) for x, y in results])
