#-*-coding:UTF-8-*-
'''
Created on 2016-8-9

@author: hanahimi
'''

from prep_2_make_datapkl import DataPackage, ImgDataset
import tensorflow as tf

from step_1_train_and_test import train_valid_test
from step_2_save_weights import saveNets_pkl
from step_3_convert_cpp import convert_cpp_models

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x,W,x_stride=1,y_stride=1, name=None):
    conv =  tf.nn.conv2d(x,W,strides=[1,x_stride,y_stride,1],padding='VALID')
    return conv

def max_pool(x, k=2, name=None):
    return tf.nn.max_pool(value=x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME', name=name)

def norm(x, lsize=4,name=None):
    return tf.nn.lrn(x,lsize,bias=10,alpha=0.001/9.0,beta=0.75,name=name)

IMSIZE = 32
OUTPUTNUM = 3
n_h = IMSIZE
n_w = IMSIZE
n_input = n_h*n_w

x = tf.placeholder("float", shape = [None,n_input])
keep_prob = tf.placeholder("float")

nets = {
    'input':"32x32, 2-labels",
    # conv1: 1x[32x32]->6x[28x28] 
    'C1w':  weight_variable([5,5,1,6]),
    'C1b':  bias_variable([6]),
    # conv2: 6x[14x14]->16x[10x10]
    'S2':   "2x2, maxpooling, 28x28->14x14",
    'C3w':  weight_variable([5,5,6,16]),
    'C3b':  bias_variable([16]),
    # conv3: 6x[5x5]->120x[1x1] 
    'S4':   "2x2, maxpooling, 10x10->5x5",
    'C5w':  weight_variable([5,5,16,120]),
    "C5b":  bias_variable([120]),
    # FC4: 120x84
    "F6" :   120,
    "F6w":  weight_variable([120,84]),
    "F6b":  bias_variable([84]),
    # FC5: 84x2
    "Fow":  weight_variable([84,OUTPUTNUM]),
    "Fob":  bias_variable([OUTPUTNUM])
}

x_image = tf.reshape(x, [-1, n_h, n_w, 1], name="x_image")
# conv1
conv1   = conv2d(x_image, nets['C1w'],name="conv1") + nets["C1b"]
relu1   = tf.nn.relu(conv1, name="relu1")
pool1   = max_pool(relu1,k=2, name="pool1")
norm1   = norm(pool1, lsize=4, name="norm1")
drop1   = tf.nn.dropout(norm1, keep_prob=keep_prob, name='drop1')
# conv2
conv2   = conv2d(drop1, nets['C3w'],name="conv2") + nets["C3b"]
relu2   = tf.nn.relu(conv2, name="relu2")
pool2   = max_pool(relu2,k=2, name="pool2")
norm2   = norm(pool2, lsize=4, name="norm2")
drop2   = tf.nn.dropout(norm2, keep_prob=keep_prob, name='drop2')
# conv3
conv3   = conv2d(drop2, nets['C5w'],name="conv3") + nets['C5b']
relu3   = tf.nn.relu(conv3, name="relu3")
pool3_flat = tf.reshape(relu3, [-1, nets["F6"]])
# FC4
fc4 = tf.nn.relu(tf.matmul(pool3_flat, nets["F6w"]) + nets["F6b"],name="fc4")
fc4_drop = tf.nn.dropout(fc4, keep_prob=keep_prob, name="fc4drop")
# FC5 - logit
logits = tf.matmul(fc4_drop,nets["Fow"])+nets["Fob"]

if __name__=="__main__":
    pass
    model_name = "leNet-full"
    data_name = "BSD_C3" # full name should be "data_name_IMSIZE"
    
    train_valid_test(x, keep_prob, logits, model_name,  data_name, IMSIZE, isTrain=True)
    
#     saveNets_pkl(nets, model_name, IMSIZE)
    
#     convert_cpp_models(model_name, IMSIZE, data_name)
    

    
    
    
    