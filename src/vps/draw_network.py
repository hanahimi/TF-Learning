#-*-coding:UTF-8-*-
'''
Created on 2017 - 9 - 27 - 4:33:50 p.m.
author: Gary-W
'''
import tensorflow as tf
from vps.posenet import Posenet
 
# 定义输入单元 placeholder
with tf.name_scope('inputs'):
    images = tf.placeholder(dtype=tf.float32, shape=[None,  224, 224, 3],name='images')
    poses_xy = tf.placeholder(dtype=tf.float32, shape=[None, 2],name='poses_xy')
    poses_ab = tf.placeholder(dtype=tf.float32, shape=[None, 2],name='poses_ab')
     
with tf.name_scope('vps'):
    weights_path = r"D:\loc_train\posenet2.npy"
    net = Posenet(images, weights_path)
 
init = tf.global_variables_initializer()
sess = tf.Session()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)
net.load_initial_weights(sess)

    
if __name__=="__main__":
    pass

