import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
tensorflow1.14 对应keras2.20
minist数据集
'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)