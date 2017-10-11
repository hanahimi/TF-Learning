#-*-coding:UTF-8-*-
"""
Created on 2017-10-10--1:28:02 p.m
author: Gary-W

learn to writer some demo functions to run posenet network

"""
import tensorflow as tf
from vps.posenet import Posenet
import numpy as np
import cv2
import random

def draw_posenework(weights_path):
    with tf.name_scope('inputs'):
        images = tf.placeholder(dtype=tf.float32, shape=[None,  224, 224, 3],name='images')
        poses_xy = tf.placeholder(dtype=tf.float32, shape=[None, 2],name='poses_xy')
        poses_ab = tf.placeholder(dtype=tf.float32, shape=[None, 2],name='poses_ab')

    with tf.name_scope('network'):
        net = Posenet(images, weights_path)
        skip_layer = ["cls1_reduction", "cls1_fc1", "cls1_fc2",
                              "cls2_reduction", "cls2_fc1", "cls2_fc2",
                              "cls3_fc"]
    init = tf.global_variables_initializer()
    sess = tf.Session()
    _writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(init)
    net.load_initial_weights(sess, skip_layer)

def draw_posenework2(weights_path):
    with tf.name_scope('inputs'):
        images = tf.placeholder(dtype=tf.float32, shape=[None,  224, 224, 3],name='images')
        poses_xy = tf.placeholder(dtype=tf.float32, shape=[None, 2],name='poses_xy')
        poses_ab = tf.placeholder(dtype=tf.float32, shape=[None, 2],name='poses_ab')

    with tf.name_scope('network'):
        net = Posenet(images, weights_path)
        skip_layer = ["cls1_reduction", "cls1_fc1", "cls1_fc2",
                              "cls2_reduction", "cls2_fc1", "cls2_fc2",
                              "cls3_fc"]
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        _writer = tf.summary.FileWriter("logs/", sess.graph)
        sess.run(init)
        net.load_initial_weights(sess, skip_layer)



def simple_testposenet(mean_path, img_path, weights_path):
    # load mean image
    image_mean = np.load(mean_path)
    print("mean image size =",image_mean.shape)

    # load test image
    image_read = cv2.imread(img_path)
    image_read = np.expand_dims(image_read, axis = 0)
    image_read = np.transpose(image_read, (0,3,1,2))
    image_norm = np.zeros(image_mean.shape, np.float32)
    image_norm[:] = image_read[:]
    image_norm -= image_mean
    print("image_norm size =",image_norm.shape)
    _,_,h,w = image_norm.shape
    h0 = (h - 224)
    w0 = (w - 224)

    with tf.name_scope('inputs'):
        images = tf.placeholder(dtype=tf.float32, shape=[None,  224, 224, 3],name='images')
          
    with tf.name_scope('vps'):
        weights_path = r"D:\loc_train\posenet4d\posenet4D_deploy_0921_0_UG_400000.npy"
        net = Posenet(images, weights_path)
    
    p3_xy = net.layers['cls3_fc_pose_xy']
    p3_ab = net.layers['cls3_fc_pose_ab']
    init = tf.global_variables_initializer()
    sess = tf.Session()
#     _writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(init)
    net.load_initial_weights(sess)
    
    # single vesion
    for i in range(5):
        hi = random.randint(0, h0)
        wi = random.randint(0, w0)
        image_test = image_norm[:,:,hi:hi+224,wi:wi+224]
        image_test = np.transpose(image_test,(0,2,3,1))
        feed = {images: image_test}
        predicted_x, predicted_q = sess.run([p3_xy, p3_ab], feed_dict=feed)
        predicted_x = np.squeeze(predicted_x)
        x, y = predicted_x[0], predicted_x[1]
        predicted_q = np.squeeze(predicted_q)
        predicted_q = predicted_q / np.linalg.norm(predicted_q)
        h.append(np.arccos(predicted_q))
        print("x =",x, " y =", y, " h =",h)
    
    # batch version
    image_test = np.zeros([5,3,224,224],np.float32)
    for i in range(5):
        hi = random.randint(0, h0)
        wi = random.randint(0, w0)
        image_test[i,:,:,:] = image_norm[0,:,hi:hi+224,wi:wi+224]
    image_test = np.transpose(image_test,(0,2,3,1))
    feed = {images: image_test}
    predicted_x, predicted_q = sess.run([p3_xy, p3_ab], feed_dict=feed)
    print("x =",predicted_x)
    print("q =",predicted_q)



if __name__=="__main__":
    pass
    weights_path = r"D:\loc_train\googlenet\version_3_enum_places\places_googlenet.npy"
    draw_posenework(weights_path)
    
    
    
