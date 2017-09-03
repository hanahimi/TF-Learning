#!/usr/bin/python
#-*-coding:utf-8-*
#########################################
  # File Name: alexnet.py
  # Author: ying chenlu
  # Mail: ychenlu92@hotmail.com 
  # Created Time: 2016/12/08,00:07:59
  # Usage: 
########################################
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import sys
import pdb
import cv2
import numpy as np
import cPickle as pickle


# converts caffe filter to tf
# tensorflow uses [filter_height, filter_width, in_channels, out_channels]
#                  2               3            1            0
# need to transpose channel axis in the weights
# caffe:  a convolution layer with 96 filters of 11 x 11 spatial dimension
# and 3 inputs the blob is 96 x 3 x 11 x 11
# caffe uses [out_channels, in_channels, filter_height, filter_width] 
#             0             1            2              3
# 
# the weights of fc layers are also been transposed

DEBUG = True
tf_layer_params_file = 'Alexnet_tf_weights.pickle'
def getInput(image_path):
    means = np.load('ilsvrc_2012_mean.npy')
    img_raw = cv2.imread(image_path) # HxWxC
    img = cv2.resize(img_raw, means.shape[1:3])
    img = np.asarray(img.transpose([2, 0, 1]), dtype=np.float32) - means # CxHxW
    return img.transpose([1, 2, 0]) # HxWxC

if not os.path.exists(tf_layer_params_file):
    import caffe
    caffe_net = '/data/yingcl/ModelZoo/CaffeNet/deploy.prototxt'
    caffe_weights = '/data/yingcl/ModelZoo/CaffeNet/bvlc_reference_caffenet.caffemodel'
    caffe_model = caffe.Net(caffe_net, caffe_weights, caffe.TEST)
    # input_img = cv2.resize(getInput('../cat.jpg'), (227, 227))
    # input_img = input_img.transpose((2, 0, 1)) # CxHxW
    # caffe_model.blobs['data'].data[0] = input_img
    # caffe_model.forward()
    # logit = caffe_model.blobs['prob'].data[0]
    # print(np.argmax(logit))
    
    tf_layer_params = {}
    for i, layer in enumerate(caffe_model.layers):
        layer_name = caffe_model._layer_names[i]
        if layer_name.startswith('conv'):
            tf_layer_params[layer_name] = {}
            weights = layer.blobs[0].data
            biases = layer.blobs[1].data
            if DEBUG == True:
                print(layer_name, ': ', weights.shape, ',', biases.shape)
            tf_layer_params[layer_name]['weights'] = weights.transpose((2, 3, 1, 0))
            tf_layer_params[layer_name]['biases'] = biases.copy()
        if layer_name.startswith("fc"):
            tf_layer_params[layer_name] = {}
            weights = layer.blobs[0].data
            biases = layer.blobs[1].data
            if DEBUG == True:
                print(layer_name, ': ', weights.shape, ',', biases.shape)
            tf_layer_params[layer_name]['weights'] = weights.transpose((1, 0))
            tf_layer_params[layer_name]['biases'] = biases.copy()
    if DEBUG == True:
        print("param shapes in tf")    
        for layer_name in tf_layer_params:
            print(layer_name, ': ', tf_layer_params[layer_name]['weights'].shape, ', ',
                    tf_layer_params[layer_name]['biases'].shape)
    with open(tf_layer_params_file, 'wb') as fp:
        pickle.dump(tf_layer_params, fp)

else:
    import tensorflow as tf

    with open(tf_layer_params_file, 'rb') as fp:
        tf_layer_params = pickle.load(fp)

    def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,\
            padding="VALID", group=1, name=None):
        '''
            k_h: kernel height
            k_w: kernel width
            c_o: channle output
        '''
        c_i = input.get_shape()[-1] # channel_input
        assert c_i%group==0
        assert c_o%group==0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        
        if group==1:
            conv = convolve(input, kernel)
        else:
            #group means we split the input  into 'group' groups along the third demention
            input_groups = tf.split(3, group, input)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
            conv = tf.concat(3, output_groups)
        # pdb.set_trace()
        return  tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list(), name=name)

    x = tf.placeholder(tf.float32, shape=(1, 227, 227, 3), name='data')
    with tf.name_scope('conv1') as scope:
        #conv1
        #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
        k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
        conv1W = tf.Variable(tf_layer_params["conv1"]['weights'], name='weights')
        conv1b = tf.Variable(tf_layer_params["conv1"]['biases'], name='biases')
        conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="VALID",
                group=1, name='conv1')
        conv1 = tf.nn.relu(conv1_in, name='relu1')

    with tf.name_scope('pool1') as scope:
        #maxpool1
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool1 = tf.nn.max_pool(conv1, ksize=[1, k_h, k_w, 1],
                strides=[1, s_h, s_w, 1], padding=padding, name='pool1')

    with tf.name_scope('lrn1') as scope:
        #lrn1
        #lrn(2, 2e-05, 0.75, name='norm1')
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn1 = tf.nn.local_response_normalization(maxpool1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name='lrn1')

    with tf.name_scope('conv2') as scope:
        #conv2
        #conv(5, 5, 256, 1, 1, group=2, name='conv2')
        k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group =2
        conv2W = tf.Variable(tf_layer_params["conv2"]['weights'], name='weights')
        conv2b = tf.Variable(tf_layer_params["conv2"]['biases'], name='biases')
        conv2_in = conv(lrn1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME",
                group=group, name='conv2')
        conv2 = tf.nn.relu(conv2_in, name='relu2')

    with tf.name_scope('pool2') as scope:
        #maxpool2
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool2 = tf.nn.max_pool(conv2, ksize=[1, k_h, k_w, 1],
                strides=[1, s_h, s_w, 1], padding=padding, name='pool2')

    with tf.name_scope('lrn2') as scope:
        #lrn2
        #lrn(2, 2e-05, 0.75, name='norm2')
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn2 = tf.nn.local_response_normalization(maxpool2 ,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name='lrn2')

    with tf.name_scope('conv3') as scope:
        #conv3
        #conv(3, 3, 384, 1, 1, name='conv3')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
        conv3W = tf.Variable(tf_layer_params["conv3"]['weights'], name='weights')
        conv3b = tf.Variable(tf_layer_params["conv3"]['biases'], name='biases')
        conv3_in = conv(lrn2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w,
                padding="SAME", group=group, name='conv3')
        conv3 = tf.nn.relu(conv3_in, name='relu3')

    with tf.name_scope('conv4') as scope:
        #conv4
        #conv(3, 3, 384, 1, 1, group=2, name='conv4')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
        conv4W = tf.Variable(tf_layer_params["conv4"]['weights'], name='weights')
        conv4b = tf.Variable(tf_layer_params["conv4"]['biases'], name='biases')
        conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w,
                padding="SAME", group=group, name='conv4')
        conv4 = tf.nn.relu(conv4_in, name='relu4')


    with tf.name_scope('conv5') as scope:
        #conv5
        #conv(3, 3, 256, 1, 1, group=2, name='conv5')
        k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv5W = tf.Variable(tf_layer_params["conv5"]['weights'], name='weights')
        conv5b = tf.Variable(tf_layer_params["conv5"]['biases'], name='biases')
        conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w,
                padding="SAME", group=group, name='conv5')
        conv5 = tf.nn.relu(conv5_in, name='relu5')

    with tf.name_scope('pool5') as scope:
        #maxpool5
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1],
                strides=[1, s_h, s_w, 1], padding=padding, name='pool5')

    with tf.name_scope('fc6') as scope:
        #fc6
        #fc(4096, name='fc6')
        maxpool5_transpose = tf.transpose(maxpool5, perm=[0, 3, 1, 2])
        fc6_input_shape = np.prod(maxpool5.get_shape().as_list()[1:])
        fc6W = tf.Variable(tf_layer_params["fc6"]['weights'], name='weights')
        fc6b = tf.Variable(tf_layer_params["fc6"]['biases'], name='biases')
        # pdb.set_trace()
        conv5_flatten = tf.reshape(maxpool5_transpose,
                [int(maxpool5.get_shape()[0]), fc6_input_shape])
        fc6 = tf.nn.relu_layer(conv5_flatten, fc6W, fc6b, name='fc6')

    with tf.name_scope('fc7') as scope:
        #fc7
        #fc(4096, name='fc7')
        fc7W = tf.Variable(tf_layer_params["fc7"]['weights'], name='weights')
        fc7b = tf.Variable(tf_layer_params["fc7"]['biases'], name='biases')
        fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b, name='fc7')

    with tf.name_scope('fc8') as scope:
        #fc8
        #fc(1000, relu=False, name='fc8')
        fc8W = tf.Variable(tf_layer_params["fc8"]['weights'], name='weights')
        fc8b = tf.Variable(tf_layer_params["fc8"]['biases'], name='biases')
        fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b, name='fc8')


    with tf.name_scope('prob') as scope:
        #prob
        #softmax(name='prob'))
        prob = tf.nn.softmax(fc8, name='prob')
    global_variables = tf.global_variables()
    for var in global_variables:
        print(var.name, var.get_shape().as_list())
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True, 
        allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # input_img = getInput('../cat.jpg')
        # input_img = cv2.resize(input_img, (227, 227))
        # logit = sess.run(prob, feed_dict={x: input_img[np.newaxis, :, :, :]})
        # print(np.argmax(logit))
        saver.save(sess, 'AlexNet.tfmodel')
