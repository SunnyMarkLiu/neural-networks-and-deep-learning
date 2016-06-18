#!D:\Miniconda2\python
# encoding: utf-8

"""
@author: MarkLiu
@file: test.py
@time: 2016/6/16 15:48
"""
import mnist_loader
import numpy as np


print 'loading dataset...'
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# training_data : (500000, 2)
# validation_data : (100000, 2)
# test_data : (100000, 2)
# print np.shape(training_data[0][0]) # (784L, 1L) 输入为列向量
# print np.shape(validation_data)
# print np.shape(test_data[0][0])

from network import Network

print 'training network...'
nn = Network([784, 20, 10])
nn.SGD(training_data, 30, 10, 0.01, validation_data)
