import tensorflow as tf

N_CLASSES = 4+10+1

class Model(object):
    @staticmethod
    def inference(x, drop_rate):
        with tf.variable_scope('hidden1'): 
            
            tf.add_to_collection('input',x)  
                   
            conv = tf.layers.conv2d(x, filters=32, kernel_size=[5, 5], padding='same')
            
            tf.add_to_collection('conv1',conv)  
            
#             norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(conv)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            
            tf.add_to_collection('pool1',pool)  
            
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden1 = dropout

        with tf.variable_scope('hidden2'):
            conv = tf.layers.conv2d(hidden1, filters=32, kernel_size=[5, 5], padding='same')
            
            tf.add_to_collection('conv2',conv) 
            
#             norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(conv)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            
            tf.add_to_collection('pool2',pool)  
              
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden2 = dropout

        with tf.variable_scope('hidden3'):
            conv = tf.layers.conv2d(hidden2, filters=48, kernel_size=[5, 5], padding='same')
            
            tf.add_to_collection('conv3',conv) 
            
#             norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(conv)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            
            tf.add_to_collection('pool3',pool)  
            
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden3 = dropout

        with tf.variable_scope('hidden4'):
            conv = tf.layers.conv2d(hidden3, filters=48, kernel_size=[5, 5], padding='same')
            
            tf.add_to_collection('conv4',conv) 
            
#             norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(conv)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            
            tf.add_to_collection('pool4',pool)  
            
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden4 = dropout
            
        with tf.variable_scope('hidden5'):
            conv = tf.layers.conv2d(hidden4, filters=96, kernel_size=[5, 5], padding='same')
            
            tf.add_to_collection('conv5',conv) 
            
#             norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(conv)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            
            tf.add_to_collection('pool5',pool)  
            
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden5 = dropout
            

        flatten = tf.reshape(hidden5, [-1, 4 * 4 * 96])
        
        tf.add_to_collection('flatten',flatten)  


        with tf.variable_scope('hidden10'):
            dense = tf.layers.dense(flatten, units=1536, activation=tf.nn.relu)
            hidden10 = dense
            
            tf.add_to_collection('hidden10',hidden10)  

        with tf.variable_scope('digit_length'):
            dense = tf.layers.dense(hidden10, units=7)
            length = dense
            
            tf.add_to_collection('length',length)  

        with tf.variable_scope('digit1'):
            dense = tf.layers.dense(hidden10, units=N_CLASSES)
            digit1 = dense
            
            tf.add_to_collection('digit1',digit1)  

        with tf.variable_scope('digit2'):
            dense = tf.layers.dense(hidden10, units=N_CLASSES)
            digit2 = dense
            
            tf.add_to_collection('digit2',digit2)  

        with tf.variable_scope('digit3'):
            dense = tf.layers.dense(hidden10, units=N_CLASSES)
            digit3 = dense
            
            tf.add_to_collection('digit3',digit3)  

        with tf.variable_scope('digit4'):
            dense = tf.layers.dense(hidden10, units=N_CLASSES)
            digit4 = dense

        with tf.variable_scope('digit5'):
            dense = tf.layers.dense(hidden10, units=N_CLASSES)
            digit5 = dense

        length_logits, digits_logits = length, tf.stack([digit1, digit2, digit3, digit4, digit5], axis=1)
        tf.add_to_collection('length_logits',length_logits)  
        return length_logits, digits_logits

    @staticmethod
    def loss(length_logits, digits_logits, length_labels, digits_labels):
        length_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=length_labels, logits=length_logits))
        digit1_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 0], logits=digits_logits[:, 0, :]))
        digit2_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 1], logits=digits_logits[:, 1, :]))
        digit3_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 2], logits=digits_logits[:, 2, :]))
        digit4_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 3], logits=digits_logits[:, 3, :]))
        digit5_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 4], logits=digits_logits[:, 4, :]))
        loss = length_cross_entropy + digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy + digit4_cross_entropy + digit5_cross_entropy
        return loss
