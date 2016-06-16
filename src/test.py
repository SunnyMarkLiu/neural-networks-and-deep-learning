#!D:\Miniconda2\python
# encoding: utf-8

"""
@author: MarkLiu
@file: test.py
@time: 2016/6/16 15:48
"""
import mnist_loader
import numpy as np


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# training_data : (500000, 2)
# validation_data : (100000, 2)
# test_data : (100000, 2)
# print np.shape(training_data[0][0]) # (784L, 1L) 输入为列向量
# print np.shape(validation_data)
# print np.shape(test_data[0][0])

# a = np.array([1,2,3,40,5,6])
# print np.argmax(a)

from network import Network

nn = Network([784, 100, 10])
nn.SGD(training_data, 10, 100, 0.01, validation_data)
