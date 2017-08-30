#-*-coding:UTF-8-*-
'''
Created on 2016-8-9

@author: hanahimi
'''
import tensorflow as tf
import numpy as np
import cv2
import time
from dataio import *

"""
function: predict
@args
    logistic TF 计算图
    x_holder
    keep_prob_holder
    x_in: 2d-Array, [num, input_size**2]
@return
    pred： 预测结果，2d-Array
"""
def predict(logistic, x_holder, x_in,keep_prob_holder, model_name, pkl_name, input_size):
    saver = tf.train.Saver()
    tf.initialize_all_variables()
    with tf.Session() as sess:
        saver.restore(sess, ("./save_ckpt/%s_%d.ckpt" % (model_name,input_size)))
        print("Model restored")
        pred = sess.run(logistic, feed_dict={x_holder: x_in, keep_prob_holder: 1.0})
        return pred

if __name__=="__main__":
    pass

    
    