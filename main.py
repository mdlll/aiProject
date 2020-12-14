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


def softmax(a):
    # 寻找数组中可能最大
    c = np.max(a)
    exp_a = np.exp(a - c)  # 防止溢出，推理在《深度学习入门xx》p228页
    exp_a_sum = np.sum(exp_a)
    y = exp_a / exp_a_sum
    return y


def identify(x):
    # 定义函数，输出等于输入
    return x


def mean_squared_error(y, t):
    # 均方误差
    # t是监督数据，y是神经网输出
    # y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    # t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    # 这个监督数据是hot-one形式表示的，则，只有数据在t[2]这个位置才是正确数据，所以要契合，y其他都要接近0，y[2]要接近1
    # 损失函数越小越接近正确解
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    # 交叉熵误差
    delta = 1e-7  # 防止为0时，logy输出负无穷
    return -np.sum(t * np.log(y + delta))


def numerical_diff(f, x):
    # 求导
    h = 1e-4  # 0.00001
    return (f(x + h) - f(f - h)) / (2 * h)


def function1(x):
    return 1e-2 * x ** 2 + 0.1 * x


def function2(x):
    return np.sum(x ** 2)


'''
机器学习使用训练数据进行学习。使用训练数据进行学习，严格来说，就是针对训练数据计算损失函数的值，找出使该值尽可能小的参数。因此，计算损失函数时必须将所有的训练数据作为对象。
也就是说，如果训练数据有 100 个的话，我们就要把这 100 个损失函数的总和作为学习的指标。

'''

if __name__ == '__main__':
    y = [[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0],[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]]
    t = [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]
    print(cross_entropy_error(np.array(y), np.array(t))/2)
