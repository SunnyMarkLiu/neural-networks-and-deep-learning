#!D:\Miniconda2\python
# encoding: utf-8

"""
@author: MarkLiu
@file: test.py
@time: 2016/6/16 15:48
"""
import mnist_loader
import numpy as np
import network2
import network

print 'loading dataset...'
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print 'training network...'

# nn = network.Network([784, 30, 10])
# nn.SGD(training_data, 30, 10, 2.0, validation_data)

nn = network2.Network([784, 30, 10])
training_cost, training_accuracy, evaluation_cost, evaluation_accuracy \
    = nn.SGD(training_data, 30, 10, 2.0,
             lmbda=1,
             validation_data=validation_data,
             monitor_training_cost=True,
             monitor_training_accuracy=True,
             monitor_evaluation_cost=True,
             monitor_evaluation_accuracy=True
             )

import matplotlib.pyplot as plt

plt.subplot(221)
plt.plot(range(0, len(evaluation_cost)), evaluation_cost, color="red")
plt.title("evaluation_cost")

plt.subplot(222)
plt.plot(range(0, len(evaluation_accuracy)), evaluation_accuracy, color="blue")
plt.title("evaluation_accuracy")

plt.subplot(223)
plt.plot(range(0, len(training_cost)), training_cost, color="black")
plt.title("training_cost")

plt.subplot(224)
plt.plot(range(0, len(training_accuracy)), training_accuracy, color="green")
plt.title("training_accuracy")
plt.show()
plt.savefig("network_evaluation.png", dip=72)
