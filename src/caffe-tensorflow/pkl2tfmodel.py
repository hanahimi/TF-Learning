#-*-coding:UTF-8-*-
'''
Created on 2017-8-31-2:38:30 pm
author: Gary-W

this script should run on python 3.5
restore the params from pkl to tensorflow graph

1. load the layer-parm dict[layer_name]["weights"] / [layer_name]["baises"]
    the data shape is:
    conv: [filter_height, filter_width, in_channels, out_channels]
    fc: [in_channels, out_channels]

2. please make sure the network structure is relative to the loaded dict

3. just use the ndarry value to initialize the varaibles of tf-graph as such way when configing
    Convolution layer or InnerProduct layer:

    example:

    with tf.name_scope('conv1') as scope:
        conv1W = conv1W = tf.Variable(tf_layer_params["conv1"]['weights'], name='weights')
        conv1b = tf.Variable(tf_layer_params["conv1"]['biases'], name='biases')
        conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="VALID",
                group=1, name='conv1')
        conv1 = tf.nn.relu(conv1_in, name='relu1')

    or
    
    with tf.name_scope('fc7') as scope:
        fc7W = tf.Variable(tf_layer_params["fc7"]['weights'], name='weights')
        fc7b = tf.Variable(tf_layer_params["fc7"]['biases'], name='biases')
        fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b, name='fc7')
    
    only some layer has Variable.

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())    # global_variables like like variable, const.
        sess.run(tf.local_variables_initializer())    # local variables like epoch_num, batch_size
        ...
        saver.save(sess, 'AlexNet.tfmodel')
TODO:
    Auto set the network for prototxt, but in the way of standard tensorflow coding way, :)
    I dont like use the network.py in git. caffe-windows, it just only suitable to caffe-style, less flexibility
    Maybe tflearn or original tf is okey enough.
'''



if __name__=="__main__":
    pass

