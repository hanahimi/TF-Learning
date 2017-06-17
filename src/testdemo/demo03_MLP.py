#-*-coding:UTF-8-*-
'''
Created on 2017年1月28日
@author: Ayumi Phoenix
使用TF定义简单的2层神经网络进行训练和测试
目标：
1. 定义计算图,并在其作用范围内:
    1.1 在计算图中定义变量和MLP模型（用变量编写矩阵计算表达式）
        使用placeholder作为输入
        使用Variable作为网络权重，并使用正态分布初始化Weight, 使用tf.constant初始化bias
    1.2 在计算图中定义损失函数loss和优化器optimizer
    1.3 定义预测计算的方法softmax
2. 定义回话执行图的计算
    2.1 变量初始化
    2.2 从端口获得数据,使用placeholder进行接受并构造feed_dict
    2.3 执行优化方法训练模型
    2.4 统计准确率
    2.5 [保存模型参数]
3. 尝试使用类来进行模型参数和结构的管理
'''
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

from util.tools import load, get_chunk

# 载入数据集
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
    def __init__(self, num_hidden, batch_size):
        """
        @num_hidden: 隐藏层的节点数量
        @batch_size：每一批处理数据的数据量
        """
        self.batch_size = batch_size
        self.test_batch_size = 500

        # Hyper Parameters
        self.num_hidden = num_hidden
        
        # Graph Related
        self.graph = tf.Graph()
        self.tf_trX = None
        self.tf_trY = None
        self.tf_tsX = None
        self.tf_tsY = None
        self.tf_ts_prediction = None
    

    def define_graph(self):
        """ 定义计算图谱 """
        with self.graph.as_default():
            # 1. 定义图谱中的各种变量
            self.tf_trX = tf.placeholder(dtype=np.float32, shape=(self.batch_size, image_size,image_size,num_channels), name='trX')
            self.tf_trY = tf.placeholder(dtype=np.float32, shape=(self.batch_size, num_labels), name='trY')
            self.tf_tsX = tf.placeholder(dtype=np.float32, shape=(self.test_batch_size, image_size,image_size,num_channels), name='tsX')
            
            # 2. 定义模型参数 W,b
            # evidence_i = sum{W_ij * x_j, j=1..n} + b_i
            # full connected layer 1 : hidden layer
            fc1_W = tf.Variable(tf.truncated_normal(shape=[image_size**2, self.num_hidden],
                                                    mean=0.0, 
                                                    stddev=0.1,
                                                    dtype=np.float32),
                                      name="fc1_W")
            fc1_b = tf.Variable(tf.constant(value=0.1,
                                            dtype=np.float32,
                                            shape=[self.num_hidden]),
                                     name="fc1_b")
            
            # full connected layer 2 : output layer
            fc2_W = tf.Variable(tf.truncated_normal(shape=[self.num_hidden, num_labels],
                                                    mean=0.0, 
                                                    stddev=0.1,
                                                    dtype=np.float32),
                                      name="fc2_W")
            fc2_b = tf.Variable(tf.constant(value=0.1, 
                                            dtype=np.float32, 
                                            shape=[num_labels]),
                                     name="fc2_b")
            
            # 2. 定义网络图谱中的计算方式(模型的设定需要在计算图的作用范围内进行)
            def model(data):
                print("create model")  
                # fully connected layer 1
                shape = data.get_shape().as_list()
                reshaped = tf.reshape(data, [shape[0], shape[1]*shape[2]*shape[3]])
                hidden = tf.nn.relu(tf.matmul(reshaped, fc1_W)+fc1_b)

                # fully connected layer 2
                output = tf.matmul(hidden, fc2_W) + fc2_b
                return output

            # 3. 定义损失函数和优化方法
            logits = model(self.tf_trX)
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits, self.tf_trY))
            # Optimizer
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(self.loss)

            # 4. Predictions for the training, validation, and test data.
            self.train_prediction = tf.nn.softmax(logits)
            logit_pred = model(self.tf_tsX)
            self.test_prediction = tf.nn.softmax(logit_pred)
            
            
    def run(self):
        """
        启动会话session，执行计算图
        """
        self.sess = tf.Session(graph=self.graph)
        with self.sess as mySess:
            tf.global_variables_initializer().run()
            ### Training
            print("starting training")
            for i, samples, labels in get_chunk(TR_SAMPLES, TR_LABELS, chunkSize=self.batch_size):
                _, l, predictions = mySess.run(
                        [self.optimizer, self.loss, self.train_prediction],
                        feed_dict={self.tf_trX: samples, self.tf_trY: labels}
                    )
                accuracy, _ = self.accuracy(predictions, labels)
                if i % 50 == 0:
                    print('Minibatch loss at step %d: %f' % (i, l))
                    print('Minibatch accuracy: %.1f%%' % accuracy)
       
            ### Testing
            accuracies = []
            confusionMatrices = []
            for i, samples, labels in get_chunk(TS_SAMPLES, TS_LABELS, chunkSize=self.test_batch_size):
                result = self.test_prediction.eval(feed_dict={self.tf_tsX: samples})
                accuracy, cm = self.accuracy(result, labels, need_confusion_matrix=False)
                accuracies.append(accuracy)
                confusionMatrices.append(cm)
                print('Test Accuracy: %.1f%%' % accuracy)
            print(' Average  Accuracy:', np.average(accuracies))
            print('Standard Deviation:', np.std(accuracies))
            # self._print_confusion_matrix(np.add.reduce(confusionMatrices))
    
    def _print_confusion_matrix(self, confusionMatrix):
        # 根据预测结果输出混淆矩阵
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
        统计预测结果，生成混淆矩阵
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
    net = Network(num_hidden=128, batch_size=1000)
    net.define_graph()
    net.run()

    