#coding:UTF-8
'''
Created on 2017年1月30日
@author: Ayumi Phoenix

使用TF定义5层的简单卷积神经网络进行训练和测试
演示卷积，激活，池化，全连接等基本操作
目标：
1. 定义计算图,并在其作用范围内:
    1.1 定义每层卷积核的权重和偏移的大小，并设定初始化方式
        权重的 shape 为 (宽度，高度，通道数，核个数)
        偏移的 shape 为 (核个数)
    1.2 [添加到TB]
        建立列表summart_list，逐个append summary
        
2 在计算图中定义CNN的计算模型
    2.1 使用conv2d为每个卷积核创建卷积层并指定每个层的输入
        - 设置每个卷积层的步长和padding类型
    2.2 为卷积层添加对应的偏移项
        conv = conv + biases
    2.2 在激活层添加在卷积层之后
    2.3 添加下采样层
    2.4 在最后加入全连接层用于分类
    2.5 使用tf.summary.image保存特征图

3 定义损失函数和优化方法
    [添加到TB]
    
4. 定义回话执行图的计算
    4.1 初始化->读数据->run(optimizer)->统计loss和accuracy
    4.2 [在run中运行summary_list]
    4.3 统计性能时采用从测试中抽取test_size个样本进行测试
        并将所有测试结果进行平均得到
'''
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

from util.tools import load, get_chunk


tr_set = load(r"../../data/ntrain_32x32.pickle")
ts_set = load(r"../../data/ntest_32x32.pickle")
TR_SAMPLES = tr_set['X']
TR_LABELS = tr_set['y']
TS_SAMPLES = ts_set['X']
TS_LABELS = ts_set['y']
print('Training set', TR_SAMPLES.shape, TR_LABELS.shape)
print('    Test set', TS_SAMPLES.shape, TS_LABELS.shape)
num_sample,image_size,_,num_channels = TR_SAMPLES.shape
_,num_labels = TR_LABELS.shape

        
class Network():
    def __init__(self, num_hidden, batch_size, conv_depth, patch_size, pooling_scale):
        """
        @num_hidden:隐藏层节点数
        @batch_size: 每个批处理的样本数
        @patch_size: 卷积核大小
        """
        self.batch_size = batch_size
        self.test_batch_size = 500
        
        # Hyper Parameters
        self.num_hidden = num_hidden
        self.patch_size = patch_size
        self.conv1_depth = conv_depth
        self.conv2_depth = conv_depth
        self.conv3_depth = conv_depth
        self.conv4_depth = conv_depth
        self.last_conv_depth = self.conv4_depth
        self.pooling_scale = pooling_scale
        self.pooling_stride = self.pooling_scale    # Max Pooling Stride

        # Graph Related
        self.graph = tf.Graph() # 创建一个空的计算图
        self.tf_trX = None
        self.tf_trY = None
        self.tf_tsX = None
        self.tf_tsY = None
        self.tf_ts_prediction = None

        # 统计
        self.merged = None
        self.train_summaries = []
        self.test_summaries = []

        self.define_graph()
        self.sess = tf.Session(graph=self.graph)
        self.writer = tf.summary.FileWriter('./board', self.graph)

        
    def define_graph(self):
        """ 定义计算图谱 """
        with self.graph.as_default():
            # 1. 定义图谱中的变量
            with tf.name_scope('inputs'):   # 对变量进行打包，使得tensorboard的图像工整一些
                self.tf_trX = tf.placeholder(dtype=np.float32, shape=(self.batch_size, image_size,image_size,num_channels), name='trX')
                self.tf_trY = tf.placeholder(dtype=np.float32, shape=(self.batch_size, num_labels), name='trY')
                self.tf_tsX = tf.placeholder(dtype=np.float32, shape=(self.test_batch_size, image_size,image_size,num_channels), name='tsX')
            
            with tf.name_scope('conv1'):
                conv1_weights = tf.Variable(
                    tf.truncated_normal([self.patch_size, self.patch_size, num_channels,self.conv1_depth],
                                         stddev=0.1))
                conv1_biases = tf.Variable(tf.zeros([self.conv1_depth]))
            
            with tf.name_scope('conv2'):
                conv2_weights = tf.Variable(
                    tf.truncated_normal([self.patch_size, self.patch_size, self.conv1_depth,self.conv2_depth],
                                         stddev=0.1))
                conv2_biases = tf.Variable(tf.zeros([self.conv2_depth]))
            
            with tf.name_scope('conv3'):
                conv3_weights = tf.Variable(
                    tf.truncated_normal([self.patch_size, self.patch_size, self.conv2_depth,self.conv3_depth],
                                         stddev=0.1))
                conv3_biases = tf.Variable(tf.zeros([self.conv3_depth]))
            
            with tf.name_scope('conv4'):
                conv4_weights = tf.Variable(
                    tf.truncated_normal([self.patch_size, self.patch_size, self.conv3_depth,self.conv4_depth],
                                         stddev=0.1))
                conv4_biases = tf.Variable(tf.zeros([self.conv4_depth]))
            
            with tf.name_scope('fc1'):
                # 在本例中，卷积操作保存输入特征的大小，缩放操作只有池化
                # 因此经过了两次池化操作后，原图像缩小的尺度为：
                down_scale = self.pooling_scale ** 2
                fc1_size = (image_size//down_scale)*(image_size//down_scale)*self.last_conv_depth
                fc1_weights = tf.Variable(
                    tf.truncated_normal([fc1_size,self.num_hidden], stddev=0.1))
                fc1_biases = tf.Variable(tf.constant(0.1, shape=[self.num_hidden]))
                
                self.train_summaries.append(tf.summary.histogram('fc1_weights', fc1_weights))
                self.train_summaries.append(tf.summary.histogram('fc1_biases', fc1_biases))
                
            with tf.name_scope('fc2'):
                fc2_weights = tf.Variable(
                    tf.truncated_normal([self.num_hidden,num_labels], stddev=0.1))
                fc2_biases = tf.Variable(tf.constant(0.1, shape=[num_labels]))
                
                self.train_summaries.append(tf.summary.histogram('fc2_weights', fc2_weights))
                self.train_summaries.append(tf.summary.histogram('fc2_biases', fc2_biases))
                
            def model(data, train=True):   
                """ 定义图谱的运算 """
                with tf.name_scope('conv1_model'):
                    with tf.name_scope('convolution'):
                        conv1 = tf.nn.conv2d(data, filter=conv1_weights,
                                             strides=[1,1,1,1],padding='SAME')
                        addition = conv1 + conv1_biases
                        hidden = tf.nn.relu(addition)
                        
                        if not train:
                            # 将激活层的输出转换为图像
                            # conv1_relu shape: (8,32,32,64)
                            # 由于该卷积层有64个filters, 因此有64幅32x32的灰度图像
                            # 8 为 batch_size 的大小，意味着每次输入8幅图像，共进行8次卷积操作
                            # 本例使用最后一幅图像进行TB记录
                            filter_map = hidden[-1]
                            filter_map = tf.transpose(filter_map, perm=[2, 0, 1])
                            filter_map = tf.reshape(filter_map, (self.conv1_depth, 32, 32, 1))
                            self.test_summaries.append(tf.summary.image('conv1_relu', tensor=filter_map, max_outputs=self.conv1_depth))
                        
                with tf.name_scope("conv2_model"):
                    with tf.name_scope('convolution'):
                        conv2 = tf.nn.conv2d(hidden, filter=conv2_weights, 
                                             strides=[1, 1, 1, 1], padding='SAME')
                        addition = conv2 + conv2_biases
                        hidden = tf.nn.relu(addition)
                        hidden = tf.nn.max_pool(
                                    hidden,
                                    ksize=[1,self.pooling_scale,self.pooling_scale,1],
                                    strides=[1,self.pooling_stride,self.pooling_stride,1],
                                    padding='SAME')
                
                with tf.name_scope('conv3_model'):
                    with tf.name_scope('convolution'):
                        conv3 = tf.nn.conv2d(hidden, filter=conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
                        addition = conv3 + conv3_biases
                    hidden = tf.nn.relu(addition)

                with tf.name_scope('conv4_model'):
                    with tf.name_scope('convolution'):
                        conv4 = tf.nn.conv2d(hidden, filter=conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
                        addition = conv4 + conv4_biases
                    hidden = tf.nn.relu(addition)
                    # if not train:
                    #     filter_map = hidden[-1]
                    #     filter_map = tf.transpose(filter_map, perm=[2, 0, 1])
                    #     filter_map = tf.reshape(filter_map, (self.conv4_depth, 16, 16, 1))
                    #     tf.image_summary('conv4_relu', tensor=filter_map, max_images=self.conv4_depth)
                    hidden = tf.nn.max_pool(
                                hidden,
                                ksize=[1,self.pooling_scale,self.pooling_scale,1],
                                strides=[1,self.pooling_stride,self.pooling_stride,1],
                                padding='SAME')

                shape = hidden.get_shape().as_list()
                reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

                with tf.name_scope('fc1_model'):
                    fc1_model = tf.matmul(reshape, fc1_weights) + fc1_biases
                    hidden = tf.nn.relu(fc1_model)

                # fully connected layer 2
                with tf.name_scope('fc2_model'):
                    return tf.matmul(hidden, fc2_weights) + fc2_biases

            # Training computation.
            logits = model(self.tf_trX)
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.tf_trY))
                self.train_summaries.append(tf.summary.scalar('Loss', self.loss))

            # Optimizer.
            with tf.name_scope('optimizer'):
                self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.001, 
                                                            rho=0.95, 
                                                            epsilon=1e-8).minimize(self.loss)

            # Predictions for the training, validation, and test data.
            with tf.name_scope('train'):
                self.train_prediction = tf.nn.softmax(logits, name='train_prediction')
            with tf.name_scope('test'):
                self.test_prediction = tf.nn.softmax(model(self.tf_tsX, train=False), name='test_prediction')

            self.merged_train_summary = tf.summary.merge(self.train_summaries)
            self.merged_test_summary = tf.summary.merge(self.test_summaries)
       
                
    def run(self):
        """ 使用 session """
        with self.sess as mySess:
            tf.global_variables_initializer().run()
            
            ### Training
            for i, samples, labels in get_chunk(TR_SAMPLES, TR_LABELS, chunkSize=self.batch_size):
                # 一次执行多个计算图节点
                _, l, predictions,summary = mySess.run(
                        [self.optimizer, self.loss, self.train_prediction, self.merged_train_summary],
                        feed_dict={self.tf_trX: samples, self.tf_trY: labels}
                    )
                self.writer.add_summary(summary, i)
                
                accuracy, _ = self.accuracy(predictions, labels)
                if i % 50 == 0:
                    print('Minibatch loss at step %d: %f' % (i, l))
                    print('Minibatch accuracy: %.1f%%' % accuracy)
       
            ### Testing
            accuracies = []
            confusionMatrices = []
            for i, samples, labels in get_chunk(TS_SAMPLES, TS_LABELS, chunkSize=self.test_batch_size):
                result, summary = mySess.run(
                    [self.test_prediction, self.merged_test_summary],
                    feed_dict={self.tf_tsX: samples}
                )
                self.writer.add_summary(summary, i)
                accuracy, cm = self.accuracy(result, labels, need_confusion_matrix=True)
                accuracies.append(accuracy)
                confusionMatrices.append(cm)
                print('Test Accuracy: %.1f%%' % accuracy)
            print(' Average  Accuracy:', np.average(accuracies))
            print('Standard Deviation:', np.std(accuracies))
            # self._print_confusion_matrix(np.add.reduce(confusionMatrices))


    # private function
    def _print_confusion_matrix(self, confusionMatrix):
        print('Confusion    Matrix:')
        for i, line in enumerate(confusionMatrix):
            print(line, line[i]/np.sum(line))
        a = 0
        for i, column in enumerate(np.transpose(confusionMatrix, (1, 0))):
            a += (column[i]/np.sum(column))*(np.sum(column)/26000)
            print(column[i]/np.sum(column),)
        print('\n',np.sum(confusionMatrix), a)
    
        
    def accuracy(self,  predictions, labels, need_confusion_matrix=False):
        '''
        计算预测的正确率与召回率
        @return: accuracy and confusionMatrix as a tuple
        '''
        _predictions = np.argmax(predictions, 1)
        _labels = np.argmax(labels, 1)
        cm = confusion_matrix(_labels, _predictions) if need_confusion_matrix else None
        # == is overloaded for numpy array
        accuracy = (100.0 * np.sum(_predictions == _labels) / predictions.shape[0])
        return accuracy, cm

if __name__=="__main__":
    pass
    net = Network(num_hidden=16, batch_size=64, patch_size=3, conv_depth=16, pooling_scale=2)
    net.run()
    