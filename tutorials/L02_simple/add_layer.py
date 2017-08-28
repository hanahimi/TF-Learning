#-*-coding:UTF-8-*-
'''
Created on 2017年8月26日-下午10:22:04
author: Gary-W
定义 一个简单的MLP 层
'''

import tensorflow as tf

def add_layer(inputs, in_size, out_size, activation_func=None):
    """ 根据输入创建一个全连接层 output = W * x + b
    """
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            # 定义该层自己的权重矩阵变量和偏移向量变量,并设定使用正态分布随机初始化
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), dtype=tf.float32, name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size])+0.1, dtype=tf.float32,name='b')   # bias 建议初始值不为1

        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        
        if activation_func is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_func(Wx_plus_b)
        
        return outputs

if __name__=="__main__":
    pass

