#!D:\Miniconda2\python
# encoding: utf-8

"""
@author: MarkLiu
@file: test.py
@time: 2016/6/16 15:48
"""
import mnist_loader
import numpy as np
from network2 import Network


# print 'loading dataset...'
# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# print 'training network...'
# nn = Network([784, 30, 10])
# nn.SGD(training_data, 30, 10, 2.0, validation_data)

print 'loading dataset...'
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print 'training network...'
nn = Network([784, 30, 10])
nn.SGD(training_data, 30, 10, 2.0, validation_data, lmbda=1)
