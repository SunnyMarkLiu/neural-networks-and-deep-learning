#!D:\Miniconda2\python
# encoding: utf-8

"""
@author: MarkLiu
@file: network.py
@time: 2016/6/26 17:25
"""
import numpy as np


class CrossEntropyCost(object):

    @staticmethod
    def cost_function(a, y):
        """
        交叉熵代价函数，解决预测错误时的 learning sloedown 问题
        np.nan_to_num : Returns an array or scalar replacing
                        Not a Number (NaN) with zero, (positive)
                        infinity with a very large number and
                        negative infinity with a very small (or negative) number
        :param a: 输出层输出
        :param y: desired output
        :return:
        """
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def output_error_delta_L(z, a, y):
        """
        输出层的错误 delta_L
        :param z: 输出层的加权输出, array
        :param a: 输出层输出（activation）, array
        :param y: desired output
        :return:
        """
        return a - y


class QuadraticCost(object):

    @staticmethod
    def cost_function(a, y):
        """
        quadratic 代价函数
        :param a: 输出层输出
        :param y:
        :return:
        """
        return 0.5 * np.sum(np.square(a - y))

    @staticmethod
    def output_error_delta_L(z, a, y):
        """
        输出层的错误 delta_L
        :param z: 输出层的加权输出, array
        :param a: 输出层输出（activation）, array
        :param y: desired output
        :return:
        """
        return (a - y) * sigmoid_derivative(z)

class Network(object):
    """
    Neural Network using CrossEntropyCost or QuadraticCost, and apply regularization
    """

    def __init__(self, sizes, cost = CrossEntropyCost, advance_init = True):
        """
        :type sizes: list
        :param sizes: 神经网络的模型，有多少层，每层有多少神经元，
                    如：Network([10, 4, 1])：该神经网络有三层，输入层（10个神经元），
                    一个隐藏层（4个神经元），一个输出层（1个神经元）
        :param cost: 使用的 cost 代价函数
        :param advance_init: 选择初始化 weights 和 biases 的函数
        """
        self.sizes = sizes
        self.layers_num = len(sizes)
        if advance_init:
            self.advance_weights_biases_init()
        else:
            self.basic_weights_biases_init()
        self.cost = cost

    def basic_weights_biases_init(self):
        """
        Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.
        :return:
        """
        # biases : 隐藏层和输出层的偏置，方便矩阵计算，每层的biase为vector
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def advance_weights_biases_init(self):
        """
        Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1 / sqrt(x).  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.
        :return:
        """
        # biases : 隐藏层和输出层的偏置，方便矩阵计算，每层的biase为vector
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

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

    def SGD(self, training_data, epochs, mini_batch_size, eta, validation_data, lmbda = 0.0):
        """
        stochastic gradient descent
        :param training_data: 训练数据，tuple的list集合，每个tuple包括(data, label)
        :param epochs: gradient descent迭代的次数
        :param mini_batch_size: 计算梯度下降的数据集大小
        :param eta: 学习速率
        :param validation_data: 验证集数据
        :param lmbda: 正则化参数，默认不正则化为0，lmbda越大惩罚越重
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
                self.update_mini_batch(mini_batch, eta, lmbda)

            # 训练完毕，计算验证集上的分类错误率
            if validation_data:
                print 'Epoch {0}: {1} / {2}'.format(i, self.classify(validation_data), len(validation_data))
            else:
                print 'Epoch {0} completed'.format(i)

    def update_mini_batch(self, mini_batch, eta, lmbda):
        """
        计算损失函数对w,b的偏导数（反相传播算法），更新w,b
        :param mini_batch: mini_batch is a list of tuples ``(x, y)``
        :param eta: learning rate
        :param lmbda: regularization param
        :return:
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            # 1. Feedforward: For each l = 2,3,...L compute activation
            a = x  # input
            activations = [x]  # 保存各层（2，3，...L）的激励
            zs = []  # 保存各层的加权输出
            for w, b in zip(self.weights, self.biases):
                z = np.dot(w, a) + b
                zs.append(z)
                a = sigmoid_activate(z)
                activations.append(a)

            # 2. Output Error delta_L 计算输出层的error
            delta = self.cost.output_error_delta_L(zs[-1], activations[-1], y)
            # 3. Backpropagate the error: For each l = L-1, L-2, ... 2
            # 保存代价函数对 w 的偏导数
            derivative_w = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
            # L与L-1层之间的对 w 的偏导数
            derivative_w[-1] = np.dot(delta, activations[-2].transpose())
            # 保存代价函数对 b 的偏导数
            derivative_b = [np.random.randn(y, 1) for y in self.sizes[1:]]
            # L与L-1层之间的对 b 的偏导数
            derivative_b[-1] = delta
            for l in xrange(2, self.layers_num):
                # noinspection PyTypeChecker
                delta = np.multiply(np.dot(self.weights[-l + 1].transpose(), delta),
                                    sigmoid_derivative(zs[-l]))
                derivative_w[-l] = np.dot(delta, activations[-l - 1].transpose())
                derivative_b[-l] = delta
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, derivative_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, derivative_w)]
        # 4. Gradient descent: For each l = L, L-1, L-2, ... 2 update weights and biases
        len_mini_batch = len(mini_batch)
        self.weights = [(1-eta*(lmbda/len_mini_batch))*w - (eta / len_mini_batch) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def classify(self, test_data):
        """
        分类
        :param test_data: tuple的list集合，每个tuple包括(data, label)
        :return:
        """
        results = [(np.argmax(self.feedforward(x)), y)
                   for x, y in test_data]
        return sum([int(x == y) for x, y in results])

    @staticmethod
    def costfunction_derivative(aL, y):
        """
        计算 cost function 对 **输出层** 激励 a 的偏导数
        C = 1/2n ∑||y(x)−y||^2 = 1/2n ∑||y(x)−aL(x)||^2
        :param aL:  output_activations
        :param y:   训练集的 label
        :return:
        """
        return aL - y


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
