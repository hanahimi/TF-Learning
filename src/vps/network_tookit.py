#-*-coding:UTF-8-*-
'''
Created on 2017-9-26-6:48:29 pm
author: Gary-W

Support functions to create networks layers
A trimmed version of caffe-to-tensorflow

reference:
http://blog.csdn.net/two_vv/article/details/76769860

'''

import tensorflow as tf

class NW():
    def __init__(self):
        pass
    
    @staticmethod
    def make_var(name, shape, trainable=True):
        '''Creates a new TensorFlow variable.'''
        return tf.get_variable(name, shape, trainable = trainable)
    
    @staticmethod
    def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,padding='SAME', groups=1, relu = True):
        # Get number of input channels
        input_channels = int(x.get_shape()[-1])

        # Create lambda function for the convolution
        convolve = lambda i, k: tf.nn.conv2d(i, k,
                                            strides = [1, stride_y, stride_x, 1],
                                            padding = padding)

        with tf.variable_scope(name) as scope:
            # Create tf variables for the weights and biases of the conv layer
            weights = tf.get_variable('weights',
                                    shape = [filter_height, filter_width,
                                    input_channels/groups, num_filters])
            biases = tf.get_variable('biases', shape = [num_filters])

            if groups == 1:
                conv = convolve(x, weights)

            # In the cases of multiple groups, split inputs & weights and
            else:
                # Split input and weights and convolve them separately
                input_groups = tf.split(axis = 3, num_or_size_splits=groups, value=x)
                weight_groups = tf.split(axis = 3, num_or_size_splits=groups, value=weights)
                output_groups = [convolve(i, k) for i,k in zip(input_groups, weight_groups)]
                # Concat the convolved output together again
                conv = tf.concat(axis = 3, values = output_groups)

            # Add biases
            bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
  
            if relu == True:
                # Apply relu function
                relu = tf.nn.relu(bias, name = scope.name)
                return relu
            else:
                return bias
                
                
    @staticmethod
    def fc(x, num_in, num_out, name, relu = True):
        with tf.variable_scope(name) as scope:
            # Create tf variables for the weights and biases
            weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
            biases = tf.get_variable('biases', [num_out], trainable=True)
    
            # Matrix multiply weights and inputs and add bias
            act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
    
            if relu == True:
                # Apply ReLu non linearity
                relu = tf.nn.relu(act)
                return relu
            else:
                return act

    @staticmethod
    def relu(x, name):
        return tf.nn.relu(x, name=name)
    
    @staticmethod
    def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
        return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                              strides = [1, stride_y, stride_x, 1],
                              padding = padding, name = name)


    @staticmethod
    def avg_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
        return tf.nn.avg_pool(x,
                              ksize=[1, filter_height, filter_width, 1],
                              strides=[1, stride_y, stride_x, 1],
                              padding=padding,
                              name=name)

    @staticmethod
    def lrn(x, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(x, depth_radius = radius,
                                                  alpha = alpha, beta = beta,
                                                  bias = bias, name = name)

    @staticmethod
    def concat(x, axis, name):
        return tf.concat(values=x, axis=axis, name=name)

    @staticmethod
    def add(x, name):
        return tf.add_n(x, name=name)

    @staticmethod
    def dropout(x, keep_prob):
        return tf.nn.dropout(x, keep_prob)

    @staticmethod
    def softmax(x, name):
        input_shape = map(lambda v: v.value, x.get_shape())
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they're not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                x = tf.squeeze(x, squeeze_dims=[1, 2])
            else:
                raise ValueError('Rank 2 tensor input expected for softmax!')
        return tf.nn.softmax(input, name)

    @staticmethod
    def batch_normalization(x, name, scale_offset=True, relu=False, variance_epsilon=1e-5):
        # NOTE: Currently, only inference is supported
        with tf.variable_scope(name) as scope:
            shape = [x.get_shape()[-1]]
            if scale_offset:
                scale = NW.make_var('scale', shape=shape)
                offset = NW.make_var('offset', shape=shape)
            else:
                scale, offset = (None, None)
            output = tf.nn.batch_normalization(
                x,
                mean = NW.make_var('mean', shape=shape),
                variance = NW.make_var('variance', shape=shape),
                offset=offset,
                scale=scale,
                variance_epsilon=variance_epsilon,
                name=name)
            if relu:
                output = tf.nn.relu(output)
        return output


if __name__=="__main__":
    pass

