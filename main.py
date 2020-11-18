import numpy as np
import matplotlib.pylab as plt


def print_hi(name):
    print(f'Hi, {name}')


def AND(x1, x2):
    n = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    theta = sum(n * w) + b
    if theta >= 0:
        print(1)
    else:
        print(0)


def step_function(x):
    y = x > 0
    print(np.array(y, dtype=np.int))
    return np.array(y, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def sortmax(a):
    # 寻找数组中可能最大
    c = np.max(a)
    exp_a = np.exp(a - c)  # 防止溢出，推理在《深度学习入门xx》p228页
    exp_a_sum = np.sum(exp_a)
    y = exp_a / exp_a_sum
    return y


def identify(x):
    #定义函数，输出等于输入
    return x


def mean_squared_error(y,t):
    #均方误差
    return 0.5*np.sum((y-t)**2)

def cross_entropy_error(y,t):
    #交叉熵误差
    delta=1e-7 # 防止为0时，logy输出负无穷
    return -np.sum(t*np.log(y+delta))


if __name__ == '__main__':
    x1 = np.array([1.0, 0.5])
    w1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    b1 = np.array([0.1, 0.2, 0.3])
    a1 = np.dot(x1, w1) + b1
    y1 = sigmoid(a1)

    w2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    b2 = np.array([0.1, 0.2])
    a2 = np.dot(y1, w2) + b2
    y2 = sigmoid(a2)

    w3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    b3 = np.array([0.1, 0.2])
    a3 = np.dot(y2, w3) + b3
    print(a3)
