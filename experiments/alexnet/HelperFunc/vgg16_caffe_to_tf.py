#!/usr/bin/python
#-*-coding:utf-8-*
#########################################
  # File Name: vgg16.py
  # Author: ying chenlu
  # Mail: ychenlu92@hotmail.com 
  # Created Time: 2016/12/11,21:38:48
  # Usage: 
########################################
'''
    1. get weights of VGG 16 from .caffemodel file,
    save them in a dict, store the dict using the cPickle module
    2. use the weights stored in the pickle file to
    initialize the variables in the tensorflow implementation
    of VGG 16, save the model using tf.Saver.
'''
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import pdb
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
model_dir = '/data/yingcl/ModelZoo/VGG/'
tf_layer_params_file = os.path.join(model_dir, 'VGG16_tf_params.pickle')
if not os.path.exists(tf_layer_params_file):
    import caffe
    caffe_net = os.path.join(model_dir, 'VGG_ILSVRC_16_layers_deploy.prototxt')
    caffe_weights = os.path.join(model_dir, 'VGG_ILSVRC_16_layers.caffemodel')
    caffe_model = caffe.Net(caffe_net, caffe_weights, caffe.TEST)

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

    batch_size = 32
    image = tf.placeholder(tf.float32, shape=[batch_size, 224, 224, 3], name='input')

    with tf.name_scope('conv1_1') as scope:
        #conv1_1
        conv1_1W = tf.Variable(tf_layer_params['conv1_1']['weights'], name='weights')
        conv1_1b = tf.Variable(tf_layer_params['conv1_1']['biases'], name='biases')
        conv1_1_in = tf.nn.bias_add(
                tf.nn.conv2d(image, conv1_1W, [1, 1, 1, 1], padding='SAME'),
                conv1_1b, name='conv_in')
        conv1_1 = tf.nn.relu(conv1_1_in, name='relu')

    with tf.name_scope('conv1_2') as scope:
        #conv1_2
        conv1_2W = tf.Variable(tf_layer_params['conv1_2']['weights'], name='weights')
        conv1_2b = tf.Variable(tf_layer_params['conv1_2']['biases'], name='biases')
        conv1_2_in = tf.nn.bias_add(
                tf.nn.conv2d(conv1_1, conv1_2W, [1, 1, 1, 1], padding='SAME'),
                conv1_2b, name='conv_in')
        conv1_2 = tf.nn.relu(conv1_2_in, name='relu')

    with tf.name_scope('pool1') as scope:
        # pool1
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool1')

    with tf.name_scope('conv2_1') as scope:
        #conv2_1
        conv2_1W = tf.Variable(tf_layer_params['conv2_1']['weights'], name='weights')
        conv2_1b = tf.Variable(tf_layer_params['conv2_1']['biases'], name='biases')
        conv2_1_in = tf.nn.bias_add(
                tf.nn.conv2d(pool1, conv2_1W, [1, 1, 1, 1], padding='SAME'),
                conv2_1b, name='conv_in')
        conv2_1 = tf.nn.relu(conv2_1_in, name='relu')

    with tf.name_scope('conv2_2') as scope:
        #conv2_2
        conv2_2W = tf.Variable(tf_layer_params['conv2_2']['weights'], name='weights')
        conv2_2b = tf.Variable(tf_layer_params['conv2_2']['biases'], name='biases')
        conv2_2_in = tf.nn.bias_add(
                tf.nn.conv2d(conv2_1, conv2_2W, [1, 1, 1, 1], padding='SAME'),
                conv2_2b, name='conv_in')
        conv2_2 = tf.nn.relu(conv2_2_in, name='relu')

    with tf.name_scope('pool2') as scope:
        # pool2
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool2')


    with tf.name_scope('conv3_1') as scope:
        #conv3_1
        conv3_1W = tf.Variable(tf_layer_params['conv3_1']['weights'], name='weights')
        conv3_1b = tf.Variable(tf_layer_params['conv3_1']['biases'], name='biases')
        conv3_1_in = tf.nn.bias_add(
                tf.nn.conv2d(pool2, conv3_1W, [1, 1, 1, 1], padding='SAME'),
                conv3_1b, name='conv_in')
        conv3_1 = tf.nn.relu(conv3_1_in, name='relu')

    with tf.name_scope('conv3_2') as scope:
        #conv3_2
        conv3_2W = tf.Variable(tf_layer_params['conv3_2']['weights'], name='weights')
        conv3_2b = tf.Variable(tf_layer_params['conv3_2']['biases'], name='biases')
        conv3_2_in = tf.nn.bias_add(
                tf.nn.conv2d(conv3_1, conv3_2W, [1, 1, 1, 1], padding='SAME'),
                conv3_2b, name='conv_in')
        conv3_2 = tf.nn.relu(conv3_2_in, name='relu')

    with tf.name_scope('conv3_3') as scope:
        #conv3_3
        conv3_3W = tf.Variable(tf_layer_params['conv3_3']['weights'], name='weights')
        conv3_3b = tf.Variable(tf_layer_params['conv3_3']['biases'], name='biases')
        conv3_3_in = tf.nn.bias_add(
                tf.nn.conv2d(conv3_2, conv3_3W, [1, 1, 1, 1], padding='SAME'),
                conv3_3b, name='conv_in')
        conv3_3 = tf.nn.relu(conv3_3_in, name='relu')

    with tf.name_scope('pool3') as scope:
        # pool3
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool3')

    with tf.name_scope('conv4_1') as scope:
        #conv4_1
        conv4_1W = tf.Variable(tf_layer_params['conv4_1']['weights'], name='weights')
        conv4_1b = tf.Variable(tf_layer_params['conv4_1']['biases'], name='biases')
        conv4_1_in = tf.nn.bias_add(
                tf.nn.conv2d(pool3, conv4_1W, [1, 1, 1, 1], padding='SAME'),
                conv4_1b, name='conv_in')
        conv4_1 = tf.nn.relu(conv4_1_in, name='relu')

    with tf.name_scope('conv4_2') as scope:
        #conv4_2
        conv4_2W = tf.Variable(tf_layer_params['conv4_2']['weights'], name='weights')
        conv4_2b = tf.Variable(tf_layer_params['conv4_2']['biases'], name='biases')
        conv4_2_in = tf.nn.bias_add(
                tf.nn.conv2d(conv4_1, conv4_2W, [1, 1, 1, 1], padding='SAME'),
                conv4_2b, name='conv_in')
        conv4_2 = tf.nn.relu(conv4_2_in, name='relu')

    with tf.name_scope('conv4_3') as scope:
        #conv4_3
        conv4_3W = tf.Variable(tf_layer_params['conv4_3']['weights'], name='weights')
        conv4_3b = tf.Variable(tf_layer_params['conv4_3']['biases'], name='biases')
        conv4_3_in = tf.nn.bias_add(
                tf.nn.conv2d(conv4_2, conv4_3W, [1, 1, 1, 1], padding='SAME'),
                conv4_3b, name='conv_in')
        conv4_3 = tf.nn.relu(conv4_3_in, name='relu')

    with tf.name_scope('pool4') as scope:
        # pool4
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool4')

    with tf.name_scope('conv5_1') as scope:
        #conv5_1
        conv5_1W = tf.Variable(tf_layer_params['conv5_1']['weights'], name='weights')
        conv5_1b = tf.Variable(tf_layer_params['conv5_1']['biases'], name='biases')
        conv5_1_in = tf.nn.bias_add(
                tf.nn.conv2d(pool4, conv5_1W, [1, 1, 1, 1], padding='SAME'),
                conv5_1b, name='conv_in')
        conv5_1 = tf.nn.relu(conv5_1_in, name='relu')

    with tf.name_scope('conv5_2') as scope:
        #conv5_2
        conv5_2W = tf.Variable(tf_layer_params['conv5_2']['weights'], name='weights')
        conv5_2b = tf.Variable(tf_layer_params['conv5_2']['biases'], name='biases')
        conv5_2_in = tf.nn.bias_add(
                tf.nn.conv2d(conv5_1, conv5_2W, [1, 1, 1, 1], padding='SAME'),
                conv5_2b, name='conv_in')
        conv5_2 = tf.nn.relu(conv5_2_in, name='relu')

    with tf.name_scope('conv5_3') as scope:
        #conv5_3
        conv5_3W = tf.Variable(tf_layer_params['conv5_3']['weights'], name='weights')
        conv5_3b = tf.Variable(tf_layer_params['conv5_3']['biases'], name='biases')
        conv5_3_in = tf.nn.bias_add(
                tf.nn.conv2d(conv5_2, conv5_3W, [1, 1, 1, 1], padding='SAME'),
                conv5_3b, name='conv_in')
        conv5_3 = tf.nn.relu(conv5_3_in, name='relu')

    with tf.name_scope('pool5') as scope:
        # pool5
        pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool5')

    with tf.name_scope('fc6') as scope:
        #fc6
        fc6_input_shape = int(np.prod(pool5.get_shape()[1:]))
        pool5_transpose = tf.transpose(pool5, perm=(0, 3, 1, 2))
        fc6W = tf.Variable(tf_layer_params['fc6']['weights'], name='weights')
        fc6b = tf.Variable(tf_layer_params['fc6']['biases'], name='biases')
        pool5_flatten = tf.reshape(pool5_transpose, [int(pool5.get_shape()[0]), fc6_input_shape])
        fc6 = tf.nn.relu_layer(pool5_flatten, fc6W, fc6b, name='relu')

    with tf.name_scope('fc7') as scope:
        #fc7
        fc7W = tf.Variable(tf_layer_params['fc7']['weights'], name='weights')
        fc7b = tf.Variable(tf_layer_params['fc7']['biases'], name='biases')
        fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b, name='relu')

    with tf.name_scope('fc8') as scope:
        #fc8
        fc8W = tf.Variable(tf_layer_params['fc8']['weights'], name='weights')
        fc8b = tf.Variable(tf_layer_params['fc8']['biases'], name='biases')
        fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b, name='relu')


    with tf.name_scope('prob') as scope:
        #prob
        prob = tf.nn.softmax(fc8, name='prob')

    global_variables = tf.global_variables()
    for var in global_variables:
        print(var.name, var.get_shape().as_list())
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, 'VGG16.tfmodel')