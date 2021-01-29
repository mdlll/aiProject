import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

'''
tensorflow1.14 对应keras2.20
线性回归代码
和保存cpkt模型
读取模型
'''
tf.compat.v1.disable_eager_execution()  # 设定非即时执行，只有在tf2.0以上才信用
# 1、创建y=2x的随机点
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3  # y=2x，但是加入了噪声
# 显示模拟数据点
# plt.plot(train_X, train_Y, 'ro', label='Original data')
# plt.legend()
# plt.show()

# 2、建立[x1,x2,x3] .* [w1, w2, w3] + b = z的训练模型, 是正向推导
X, Y = tf.compat.v1.placeholder('float', ), tf.compat.v1.placeholder('float', )  # 创建占位符
W, b = tf.Variable(tf.compat.v1.random_normal([1]), name='weight'), tf.Variable(tf.zeros([1]), name='bias')  # 创建模型参数
z = tf.multiply(X, W) + b  # 模型结构

# 3、建立反向理论
cost = tf.reduce_mean(
    tf.square(Y - z))  # 定义一个cost，它等于生成值与真实值的平方差。 computes the mean of elements across dimensions of a tensor.
learing_rate = 0.010
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learing_rate).minimize(
    cost)  # GradientDescentOptimizer函数是一个封装好的梯度下降算法，cost

# 4、 使用seesion进行训练
init = tf.compat.v1.global_variables_initializer()  # 初始化所有变量
trainning_epochs = 20
display_step = 2
saver = tf.compat.v1.train.Saver(max_to_keep=1)  # 设置保存模型,max表示只保留一个，在执行保存点的时候
saver_dir = '.\\log\\'

with tf.compat.v1.Session() as sess:
# 这里有个bug是，这个程序在这里执行保存失败，不知道为什么，但是换个project就可以，就，很迷惑
    sess.run(init)
    plotdata = {"batchsize": [], "loss": []}  # 存放批次值和损失值
    for epoch in range(trainning_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
            # 显示训练中的详细信息
            if epoch % display_step == 0:
                loss = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
                print("Epoch:", epoch + 1, "cost=", loss, "W=", sess.run(W), "b=", sess.run(b))
                if not (loss == "NA"):
                    plotdata["batchsize"].append(epoch)
                    plotdata["loss"].append(loss)
    print("Finished!")
    saver.save(sess, save_path=saver_dir + 'line_model.cpkt')  # 保存模型
    print("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}), "W=", sess.run(W), "b=", sess.run(b))
    print("x=0.2，z=", sess.run(z, feed_dict={X: 0.2}))

# 读取模型
with tf.compat.v1.Session() as sess2:
    sess2.run(tf.global_variables_initializer())
    saver.restore(sess2, save_path=saver_dir + 'line_model.cpkt')
    print('x = 0.2, z= ', sess2.run(z, feed_dict={X: 0.2}))

print_tensors_in_checkpoint_file(saver_dir + "line_model.cpkt", None, True)  # 打印模型结构
