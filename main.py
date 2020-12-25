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
    # 求导 由于h不可以无限接近0，so (f(x+h)-f(x))/h 前向差分, 会比下面的返回方式差，下面称中心差分
    h = 1e-4  # 0.00001
    a = f(x + h)  # 如果写成 (f(x+h)-f(x-h)/(2*h), 执行f(x+h)后，f会赋值成浮点数，报错
    b = f(x - h)
    return (a - b) / (2 * h)


def function1(x):
    # 数值微分的例子
    return 1e-2 * x ** 2 + 0.1 * x


def function2(x):
    # such as: y=x1**2+x2**2
    return np.sum(x ** 2)


def numerical_gradient_no_batch(f, x):
    # 偏导数，计算单个
    h = 1e-4
    grad = np.zeros_like(x)  # 生成长度为x的数组
    for idx in range(x.size):
        val = x[idx]
        x[idx] = val + h
        diff1 = f(x)  # 所有数字都放进去
        x[idx] = val - h
        diff2 = f(x)
        grad[idx] = (diff1 - diff2) / (2 * h)
        x[idx] = val
    return grad


def numerical_gradient_all(f, x):
    if x.ndim == 1:
        return numerical_gradient_no_batch(f, x)
    else:
        grad = np.zeros_like(x)
        for id, xi in enumerate(x):  # 使用枚举的方式才有id可以获取
            grad[id] = numerical_gradient_no_batch(f, xi)

    return grad


def gradient_descent(f, init_x, step=100, lr=0.1):
    # 梯度法求f的最小值
    x = init_x

    for xi in range(step):
        # 导数为正，则向前就会增加，所以x=x-lr*numberical_gradient_all
        # lr 是学习率，也叫超参数，太大太小都影响学习效果的
        x -= lr * numerical_gradient_all(f, x)

    return x


'''
机器学习使用训练数据进行学习。使用训练数据进行学习，严格来说，就是针对训练数据计算损失函数的值，找出使该值尽可能小的参数。因此，计算损失函数时必须将所有的训练数据作为对象。
也就是说，如果训练数据有 100 个的话，我们就要把这 100 个损失函数的总和作为学习的指标。
'''

if __name__ == '__main__':
    x = np.array([3.0, 4.0])
    xx = gradient_descent(function2, x)
    # 结果非常接近最小值0
    print(xx)

if __name__ == '__main__':
    ## example 1

    x = np.linspace(0, 10, 40)
    y = x ** 2 * np.exp(-x)

    u = np.array([x[i + 1] - x[i] for i in range(len(x) - 1)])
    v = np.array([y[i + 1] - y[i] for i in range(len(x) - 1)])

    x = x[:len(u)]  # 使得维数和u,v一致
    y = y[:len(v)]

    c = np.randn(len(u))  # arrow颜色

    plt.figure()
    plt.quiver(x, y, u, v, c, angles='xy', scale_units='xy', scale=1)  # 注意参数的赋值
    np.show()

if __name__ == '__main__':
    x = np.arange(-2, 2, 0.2)
    y = np.arange(-2, 2, 0.2)
    X, Y = np.meshgrid(x, y)

    X = X.flatten()
    Y = Y.flatten()
    grad = numerical_gradient_all(function2, np.array([X, Y]))
    # when x0=-2,x1=-2, grad to x0, grad(f/x0)=g[0][0]
    x0 = -2 - grad[0][0]
    print(x0)
    # when x0=0,x1=0, grad to x0, grad(f/x0)=g[0][199]
    x1 = 0 - grad[0][200]
    print(max(grad[1]))
    # print(grad[0],'\n',grad[1],x0)

    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1], color='#666666', angles='xy')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    # plt.draw()
    # plt.show()

if __name__ == '__main__':
    y = [[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0], [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]]
    t = [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]
    # print(cross_entropy_error(np.array(y), np.array(t))/2)
