import tensorflow as tf


# keep the size of conv-map as same as input
DEFAULT_PADDING = 'SAME'

N_CLASSES = 4+10+1   # 0123456789ABEF*

def make_var(name, shape):
    '''Creates a new TensorFlow variable.'''
    return tf.get_variable(name, shape)

def hiddenlayer(name, 
                input, 
                k_h, k_w, c_o,
                s_h,s_w,
                biased=True,
                relu=True,
                padding=DEFAULT_PADDING,
                norm=False,
                dropout_rate = 0.5,
                pooling = "max_pooling"):
    
    c_i = input.get_shape()[-1]
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    with tf.variable_scope(name):
        tf.add_to_collection('input', input)
        weights = make_var("weights", [k_h, k_w, c_i, c_o])
        conv = convolve(input, weights)
        tf.summary.histogram(name+"/weights", weights)
        
        if biased:
            biases = make_var('biases', [c_o])
            conv = tf.nn.bias_add(conv, biases)
            tf.summary.histogram(name+"/biases", biases)
            
        tf.add_to_collection(name,conv)
        
        if norm == True:
            conv = tf.layers.batch_normalization(conv)
        if relu == True:
            activation = tf.nn.relu(conv)
        if pooling=="max_pooling":
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding=DEFAULT_PADDING)
        elif pooling=="avg_pooling":
            pool = tf.layers.average_pooling2d(activation, pool_size=[2, 2], strides=2, padding=DEFAULT_PADDING)
        else:
            pool = activation

        if 0 < dropout_rate < 1:
            dropout = tf.layers.dropout(pool, rate=dropout_rate)
        else:
            dropout = pool
        
        return dropout

def denselayer(name, input, num_out, relu=False):
    with tf.variable_scope(name) as scope:
        input_shape = input.get_shape()
        if input_shape.ndims == 4:
            # The input is spatial. Vectorize it first.
            dim = 1
            for d in input_shape[1:].as_list():
                dim *= d
            feed_in = tf.reshape(input, [-1, dim])
        else:
            feed_in, dim = (input, input_shape[-1].value)
        weights = make_var('weights', shape=[dim, num_out])
        biases = make_var('biases', [num_out])
        tf.summary.histogram(name+"/weights", weights)
        tf.summary.histogram(name+"/biases", biases)
        
        op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
        fc = op(feed_in, weights, biases, name=scope.name)
        
        tf.add_to_collection(name, fc)
        
        return fc

class Model(object):
    @staticmethod
    def inference(x, drop_rate):
        
        h1 = hiddenlayer('hidden1', x, 5,5,32,1,1,
                         biased=True,
                         relu=True,
                         padding=DEFAULT_PADDING,
                         dropout_rate = drop_rate,
                         pooling = "max_pooling")

        h2 = hiddenlayer('hidden2', h1, 5,5,32,1,1,
                         biased=True,
                         relu=True,
                         padding=DEFAULT_PADDING,
                         dropout_rate = drop_rate,
                         pooling = "max_pooling")

        h3 = hiddenlayer('hidden3', h2, 5,5,48,1,1,
                         biased=True,
                         relu=True,
                         padding=DEFAULT_PADDING,
                         dropout_rate = drop_rate,
                         pooling = "max_pooling")

        h4 = hiddenlayer('hidden4', h3, 3,3,48,1,1,
                         biased=True,
                         relu=True,
                         padding=DEFAULT_PADDING,
                         dropout_rate = drop_rate,
                         pooling = "max_pooling")

        h5 = hiddenlayer('hidden5', h4, 3,3,96,1,1,
                         biased=True,
                         relu=True,
                         padding=DEFAULT_PADDING,
                         dropout_rate = drop_rate,
                         pooling = "max_pooling")
        
        h10 = denselayer("hidden10", h5, num_out=1536)

        char_length = denselayer("char_length", h10, num_out=7)
        
        char01 = denselayer("char01", h10, num_out=N_CLASSES)
        char02 = denselayer("char02", h10, num_out=N_CLASSES)
        char03 = denselayer("char03", h10, num_out=N_CLASSES)
        char04 = denselayer("char04", h10, num_out=N_CLASSES)
        char05 = denselayer("char05", h10, num_out=N_CLASSES)
        
        length_logits, chars_logits = char_length, tf.stack([char01, char02, char03, char04, char05], axis=1)
        return length_logits, chars_logits


    @staticmethod
    def loss(length_logits, chars_logits, length_labels, chars_labels):
        length_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=length_labels, logits=length_logits))
        char0_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=chars_labels[:, 0], logits=chars_logits[:, 0, :]))
        char1_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=chars_labels[:, 1], logits=chars_logits[:, 1, :]))
        char2_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=chars_labels[:, 2], logits=chars_logits[:, 2, :]))
        char3_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=chars_labels[:, 3], logits=chars_logits[:, 3, :]))
        char4_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=chars_labels[:, 4], logits=chars_logits[:, 4, :]))
        
        loss = length_cross_entropy + char0_cross_entropy + char1_cross_entropy + char2_cross_entropy + char3_cross_entropy + char4_cross_entropy
        return loss
