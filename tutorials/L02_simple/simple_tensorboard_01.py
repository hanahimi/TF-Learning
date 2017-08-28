#-*-coding:UTF-8-*-
'''
Created on 2017年8月27日-下午3:16:47
author: Gary-W
使用tensorboard对建立的网络进行可视化
keyword:
1 为每个op添加名字
2 使用命名域包裹对应的模块便于理清网络结构
3 在session 后面定义writer
4 最后在terminal（终端 这里使用conda prompt）中 ，使用以下命令：
    activate tensorflow
    cd 到 logs 所在的目录
    tensorboard --logdir logs
这时会显示 Starting TensorBoard b'54' at http://DESKTOP-HS7PMTL:6006
将终端中输出的网址复制到Google Chrome浏览器中，便可以看到之前定义的视图框架了

'''
import tensorflow as tf
import numpy as np
from L02_simple.add_layer import add_layer


# 定义输入数据
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0.0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# 定义输入单元 placeholder
with tf.name_scope('inputs'):
    xs = tf.placeholder(dtype=tf.float32, shape=[None, 1],name='x_input')
    ys = tf.placeholder(dtype=tf.float32, shape=[None, 1],name='y_input')

# 定义一个3层的nn
hidden_size = 10
l1 = add_layer(inputs=xs, in_size=1, out_size=hidden_size, activation_func=tf.nn.relu)  # 隐藏层
prediction = add_layer(l1, hidden_size, 1, activation_func=None)

# 定义代价函数和训练参数
with tf.name_scope('loss'):
    loss_unit = tf.reduce_sum(tf.square(ys-prediction), reduction_indices = [1])    # 按维数求和 平方差
    loss = tf.reduce_mean(loss_unit, name='loss')    # 按数据量求均值 均方差
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)

# 
# for i in range(1000):
#     _,loss_value = sess.run([train_step, loss], feed_dict={xs:x_data, ys:y_data})
#     if i % 50 == 0:
#         print("loss = ",loss_value)
#         prediction_value = sess.run(prediction, feed_dict={xs:x_data})

if __name__=="__main__":
    pass

