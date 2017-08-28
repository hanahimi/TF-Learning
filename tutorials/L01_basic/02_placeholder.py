#-*-coding:UTF-8-*-
'''
Created on 2017年8月9日-下午4:54:11
author: Gary-W
'''

import tensorflow as tf
import numpy as np

x1 = tf.placeholder(dtype=tf.float32, shape=None, name="x1")
x2 = tf.placeholder(dtype=tf.float32, shape=None, name="x2")
y1 = x1 * x2

# elements-wise muliply broadcast into biggest dim
if 1:
    # horizon vector
    x3 = tf.placeholder(dtype=tf.float32, shape=[1,2], name="x3")
    x4 = tf.placeholder(dtype=tf.float32, shape=[1,2], name="x4")
    y2 = tf.matmul(x3, x4)     # =x3*x4, = tf.multiply(x3, x4)
    v3 = [[1,2]]
    v4 = [[3,4]]
else:
    # vertial vector
    x3 = tf.placeholder(dtype=tf.float32, shape=[2,1], name="x3")
    x4 = tf.placeholder(dtype=tf.float32, shape=[2,1], name="x4")
    y2 = tf.matmul(x3, x4)     
    v3 = [[1],[2]]
    v4 = [[3],[4]]

# matrix muliply
m1 = tf.placeholder(dtype=tf.float32, shape=[1,2], name="m1")
m2 = tf.placeholder(dtype=tf.float32, shape=[2,1], name="m2")
m1v = np.array([[5,6]])
m2v = np.array([[2],[3]])
m3_1x1 = tf.matmul(m1, m2)
m3_2x2 = tf.matmul(m2, m1)


with tf.Session() as sess:
    x1_val, x2_val, y1_val = sess.run([x1, x2, y1], 
                                      feed_dict = {x1:3, x2:2})
    print(x1_val,"*", x2_val,"=", y1_val,"\n")
    
    x3_val, x4_val, y2_val = sess.run([x3, x4, y2], 
                                      feed_dict = {x3:v3, x4:v4})
    print(x3_val,"\n*\n", x4_val,"\n=\n", y2_val,"\n")
    
    m1_val, m2_val, m3_1x1_val,m3_2x2_val = sess.run([m1,m2,m3_1x1,m3_2x2], 
                                      feed_dict = {m1:m1v, m2:m2v})
    print(m1_val,"\n*\n", m2_val,"\n=\n", m3_1x1_val,"\n")
    print(m2_val,"\n*\n", m1_val,"\n=\n", m3_2x2_val,"\n")
    
    
if __name__=="__main__":
    pass

