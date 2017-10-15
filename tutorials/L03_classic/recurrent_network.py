#-*-coding:UTF-8-*-
'''
Created on 2017年8月26日-下午10:22:04
author: Gary-W
Recurrent Neural Network.
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
    [RNN/LSTM循环神经网络长短期记忆网络使用] https://my.oschina.net/yilian/blog/665412
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/

这次我们会使用 RNN 来进行分类的训练 (Classification). 
会继续使用到手写数字 MNIST 数据集. 
让 RNN 从每张图片的第一行像素读到最后一行, 然后再进行分类判断
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(1)

# 导入数据
mnist = input_data.read_data_sets(r'D:\Ayumi\workspace\dataset\character_number\MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001                  # learning rate
training_iters = 100000     # train step 上限
batch_size = 128            
# MNIST data input (img shape: 28*28)
n_inputs = 28               # 按行扫描输入，每次输入一行数据（共28列） 
n_steps = 28                # time steps，每幅图像28行，因此序列长度为28
n_hidden_units = 128        # neurons in hidden layer
n_classes = 10              # MNIST classes (0-9 digits)

# x y placeholder
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# 对 weights biases 初始值的定义
weights = {
    
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),    # shape (28, 128)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))   # shape (128, 10)
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),  # shape (128, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))       # shape (10, )
}

# 接着开始定义 RNN 主体结构, 这个 RNN 总共有 3 个组成部分
#  ( input_layer, cell, output_layer).
def RNN(X, weights, biases):
    # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
    # X ==> (128 batches * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # X_in = W*X + b
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batches, 28 steps, 128 hidden) 换回3维
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # 使用 basic LSTM Cell.
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32) # 初始化全零 state
    
    """
    如果使用tf.nn.dynamic_rnn(cell, inputs), 我们要确定 inputs 的格式. 
    tf.nn.dynamic_rnn 中的 time_major 参数会针对不同 inputs 格式有不同的值.
    """
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, 
                                             X_in, 
                                             initial_state=init_state, 
                                             time_major=False)
    
    """
    最后是 output_layer 和 return 的值. 因为这个例子的特殊性, 有两种方法可以求得 results.
    """
#     # 方式一: 直接调用final_state 中的 h_state (final_state[1]) 来进行运算:
#     results = tf.matmul(final_state[1], weights['out']) + biases['out']

    # 方式二: 调用最后一个 outputs (在这个例子中,和上面的final_state[1]是一样的):
    # 把 outputs 变成 列表 [(batch, outputs)..] * steps
    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    #选取最后一个 output
    
    return results


# 定义好了 RNN 主体结构后, 我们就可以来计算 cost 和 train_op:
pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
        }))
        step += 1

if __name__=="__main__":
    pass











