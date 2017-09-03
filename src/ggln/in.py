#-*-coding:UTF-8-*-
'''
Created on 2017年8月7日-上午11:56:30
author: Gary-W
'''
import tensorflow as tf

def inception_unit(inputdata, weights, biases):
    # A3 inception 3a
    inception_in = inputdata
    
    # Conv 1x1+S1
    inception_1x1_S1 = tf.nn.conv2d(inception_in, weights['inception_1x1_S1'], strides=[1,1,1,1], padding='SAME')
    inception_1x1_S1 = tf.nn.bias_add(inception_1x1_S1, biases['inception_1x1_S1'])
    inception_1x1_S1 = tf.nn.relu(inception_1x1_S1)
    # Conv 3x3+S1
    inception_3x3_S1_reduce = tf.nn.conv2d(inception_in, weights['inception_3x3_S1_reduce'], strides=[1,1,1,1], padding='SAME')
    inception_3x3_S1_reduce = tf.nn.bias_add(inception_3x3_S1_reduce, biases['inception_3x3_S1_reduce'])
    inception_3x3_S1_reduce = tf.nn.relu(inception_3x3_S1_reduce)
    inception_3x3_S1 = tf.nn.conv2d(inception_3x3_S1_reduce, weights['inception_3x3_S1'], strides=[1,1,1,1], padding='SAME')
    inception_3x3_S1 = tf.nn.bias_add(inception_3x3_S1, biases['inception_3x3_S1'])
    inception_3x3_S1 = tf.nn.relu(inception_3x3_S1)
    # Conv 5x5+S1
    inception_5x5_S1_reduce = tf.nn.conv2d(inception_in, weights['inception_5x5_S1_reduce'], strides=[1,1,1,1], padding='SAME')
    inception_5x5_S1_reduce = tf.nn.bias_add(inception_5x5_S1_reduce, biases['inception_5x5_S1_reduce'])
    inception_5x5_S1_reduce = tf.nn.relu(inception_5x5_S1_reduce)
    inception_5x5_S1 = tf.nn.conv2d(inception_5x5_S1_reduce, weights['inception_5x5_S1'], strides=[1,1,1,1], padding='SAME')
    inception_5x5_S1 = tf.nn.bias_add(inception_5x5_S1, biases['inception_5x5_S1'])
    inception_5x5_S1 = tf.nn.relu(inception_5x5_S1)
    # MaxPool
    inception_MaxPool = tf.nn.max_pool(inception_in, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
    inception_MaxPool = tf.nn.conv2d(inception_MaxPool, weights['inception_MaxPool'], strides=[1,1,1,1], padding='SAME')
    inception_MaxPool = tf.nn.bias_add(inception_MaxPool, biases['inception_MaxPool'])
    inception_MaxPool = tf.nn.relu(inception_MaxPool)
    # Concat
    #tf.concat(concat_dim, values, name='concat')
    #concat_dim 是 tensor 连接的方向（维度）， values 是要连接的 tensor 链表， name 是操作名。 cancat_dim 维度可以不一样，其他维度的尺寸必须一样。
    inception_out = tf.concat(concat_dim=3, values=[inception_1x1_S1, inception_3x3_S1, inception_5x5_S1, inception_MaxPool])
    return inception_out


if __name__=="__main__":
    pass

