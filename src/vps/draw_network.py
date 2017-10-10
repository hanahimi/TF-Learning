#-*-coding:UTF-8-*-
'''
Created on 2017 - 9 - 27 - 4:33:50 p.m.
author: Gary-W
'''
import tensorflow as tf
from vps.posenet import Posenet
import numpy as np
import cv2
import random
# load mean image
mean_path = r"D:\loc_train\posenet4d\testdata\image_mean.npy"
image_mean = np.load(mean_path)
print("mean image size =",image_mean.shape)

# load test image
img_path = r"D:\loc_train\posenet4d\testdata\002488.jpg"

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

# 定义输入单元 placeholder
with tf.name_scope('inputs'):
    images = tf.placeholder(dtype=tf.float32, shape=[None,  224, 224, 3],name='images')
    poses_xy = tf.placeholder(dtype=tf.float32, shape=[None, 2],name='poses_xy')
    poses_ab = tf.placeholder(dtype=tf.float32, shape=[None, 2],name='poses_ab')
      
with tf.name_scope('vps'):
    weights_path = r"D:\loc_train\posenet4d\posenet4D_deploy_0921_0_UG_400000.npy"
    net = Posenet(images, weights_path)

p3_xy = net.layers['cls3_fc_pose_xy']
p3_ab = net.layers['cls3_fc_pose_ab']

init = tf.global_variables_initializer()
sess = tf.Session()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)
net.load_initial_weights(sess)
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

