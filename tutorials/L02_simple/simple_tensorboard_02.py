#coding:UTF-8-*-
'''
Created on 2017年8月27日-下午4:14:08
author: Gary-W
1 在 layer 中为 Weights, biases 设置变化图表 
    tf.summary.histogram
2 设置loss的变化图 EVENT 
    tf.summary.scalar
3 在Session后给所有训练图‘合并‘
    merged = tf.summary.merge_all()
'''
import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, n_layer, activation_func=None):
    """ 根据输入创建一个全连接层 output = W * x + b
    """
    layer_name='layer%s' % n_layer  ## define a new var
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            # 定义该层自己的权重矩阵变量和偏移向量变量,并设定使用正态分布随机初始化
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), dtype=tf.float32, name='W')
            tf.summary.histogram(layer_name+"/weights", Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size])+0.1, dtype=tf.float32,name='b')   # bias 建议初始值不为1
            tf.summary.histogram(layer_name+"/biases", biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        
        if activation_func is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_func(Wx_plus_b)
        tf.summary.histogram(layer_name+"/outputs", outputs)
        return outputs

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
l1 = add_layer(inputs=xs, in_size=1, out_size=hidden_size, n_layer = 1,activation_func=tf.nn.relu)  # 隐藏层
prediction = add_layer(l1, hidden_size, 1, n_layer = 2, activation_func=None)

# 定义代价函数和训练参数
with tf.name_scope('loss'):
    loss_unit = tf.reduce_sum(tf.square(ys-prediction), reduction_indices = [1])    # 按维数求和 平方差
    loss = tf.reduce_mean(loss_unit, name='loss')    # 按数据量求均值 均方差
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
# 先对所有的图进行打包，然后写到log中
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)

sess.run(init)

 
for i in range(1000):
    _,loss_value = sess.run([train_step, loss], feed_dict={xs:x_data, ys:y_data})
    if i % 50 == 0:
        # 可以记录每次的训练结果
        rs = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(rs, i)
        
        print("loss = ",loss_value)
        prediction_value = sess.run(prediction, feed_dict={xs:x_data})



if __name__=="__main__":
    pass

