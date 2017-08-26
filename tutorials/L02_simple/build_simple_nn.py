#-*-coding:UTF-8-*-
'''
Created on 2017年8月26日-下午10:22:04
author: Gary-W
使用add_layer建立一个简单的nn
并进行结果可视化（动态效果）
'''
import tensorflow as tf
import numpy as np
from L02_simple.add_layer import add_layer
import matplotlib.pyplot as plt

# 定义输入数据
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0.0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# 定义输入单元
xs = tf.placeholder(dtype=tf.float32, shape=[None, 1])
ys = tf.placeholder(dtype=tf.float32, shape=[None, 1])

# 定义一个3层的nn
hidden_size = 10
l1 = add_layer(inputs=xs, in_size=1, out_size=hidden_size, activation_func=tf.nn.relu)  # 隐藏层
prediction = add_layer(l1, hidden_size, 1, activation_func=None)

# 定义代价函数和训练参数
loss_unit = tf.reduce_sum(tf.square(ys-prediction), reduction_indices = [1])    # 按维数求和 平方差
loss = tf.reduce_mean(loss_unit)    # 按数据量求均值 均方差
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
 
fig =  plt.figure()
_d_ = 0.5
plt.ylim((np.min(y_data)-_d_, np.max(y_data)+_d_))
ax = fig.add_subplot(1,1,1) # 使用动画效果
ax.scatter(x_data, y_data,  c='blue', marker='.')
plt.ion()   # 设置show以后不暂停 py3.5
plt.show(block=False)
# plt.show(block=False)    # 就版本 matplotlib不暂停

lines = None    # 防止ide出现错误提示（不加也可以）
for i in range(1000):
    _,loss_value = sess.run([train_step, loss], feed_dict={xs:x_data, ys:y_data})
    if i % 50 == 0:
        print("loss = ",loss_value)
        prediction_value = sess.run(prediction, feed_dict={xs:x_data})

        # 把prediction的值画到现有的图片上
#         lines = ax.plot(x_data, prediction_value, 'r-', lw=2)
#         # 暂停一定时间
#         plt.pause(0.1)
#         # 去除掉现有的划线以便刷新
#         ax.lines.remove(lines[0])

        # 先抹除再描画效果会好点
        try:
            ax.lines.remove(lines[0])   # 这里ide会报错但是不会有编译问题,因为在下一次循环时 line就会被定义
        except Exception:
            pass
        lines = ax.plot(x_data, prediction_value, 'r-', lw=2)
        plt.pause(0.1)

plt.pause(-1)
        
if __name__=="__main__":
    pass


