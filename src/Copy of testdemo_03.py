#-*-coding:UTF-8-*-
'''
Created on 2017年1月28日
@author: Ayumi Phoenix
'''
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

from mytools import load


tr_set = load(r"data/ntrain_32x32.pickle")
ts_set = load(r"data/ntest_32x32.pickle")
TR_SAMPLES = tr_set['X']
TR_LABELS = tr_set['y']
TS_SAMPLES = ts_set['X']
TS_LABELS = ts_set['y']
print('Training set', TR_SAMPLES.shape, TR_LABELS.shape)
print('    Test set', TS_SAMPLES.shape, TS_LABELS.shape)

num_sample,image_size,_,num_channels = TR_SAMPLES.shape
_,num_labels = TR_LABELS.shape

def get_chunk(samples, labels, chunkSize):
    '''
    Iterator/Generator: get a batch of data
    这个函数是一个迭代器/生成器，用于每一次只得到 chunkSize 这么多的数据
    用于 for loop， just like range() function
    '''
    if len(samples) != len(labels):
        raise Exception('Length of samples and labels must equal')
    stepStart = 0    # initial step
    i = 0
    while stepStart < len(samples):
        stepEnd = stepStart + chunkSize
        if stepEnd < len(samples):
            yield i, samples[stepStart:stepEnd], labels[stepStart:stepEnd]
            i += 1
        stepStart = stepEnd
        
class Network():
    def __init__(self, num_hidden, batch_size):
        """
        @num_hidden:隐藏层节点数
        @batch_size: 每个批处理的样本数
        """
        self.batch_size = batch_size
        self.test_batch_size = 500

        # Hyper Parameters
        self.num_hidden = num_hidden
        
        # Graph Related
        self.graph = tf.Graph() # 创建一个空的计算图
        self.tf_trX = None
        self.tf_trY = None
        self.tf_tsX = None
        self.tf_tsY = None
        self.tf_ts_prediction = None
    

    def define_graph(self):
        """ 定义计算图谱 """
        with self.graph.as_default():
            # 1. 定义图谱中的变量
            self.tf_trX = tf.placeholder(dtype=np.float32, shape=(self.batch_size, image_size,image_size,num_channels), name='trX')
            self.tf_trY = tf.placeholder(dtype=np.float32, shape=(self.batch_size, num_labels), name='trY')
            self.tf_tsX = tf.placeholder(dtype=np.float32, shape=(self.test_batch_size, image_size,image_size,num_channels), name='tsX')
            
            # evidence_i = sum{W_ij * x_j, j=1..n} + b_i
            # full connected layer 1 -- hidden layer
            # 从高斯分布中随机初始化权重作为变量的初值
#             fc1_W = tf.Variable(tf.truncated_normal(shape=[image_size**2, self.num_hidden],mean=0.0, stddev=0.1,dtype=np.float32),
#                                       name="fc1_W")
#             # 用常数0.1初始化权重偏移量
#             fc1_b = tf.Variable(tf.constant(value=0.1,dtype=np.float32,shape=[self.num_hidden]),
#                                      name="fc1_b")
#             
#             # full connected layer 2 -- output layer
#             fc2_W = tf.Variable(tf.truncated_normal(shape=[self.num_hidden, num_labels],mean=0.0, stddev=0.1,dtype=np.float32),
#                                       name="fc2_W")
#             fc2_b = tf.Variable(tf.constant(value=0.1, dtype=np.float32, shape=[num_labels]),
#                                      name="fc2_b")
            fcW = []
            fcb = []
            for i in range(len(self.num_hidden)):
                if i == 1:
                    _fcW = tf.Variable(tf.truncated_normal(shape=[image_size**2, self.num_hidden[i]],mean=0.0, stddev=0.1,dtype=np.float32),
                                  name="fc%d_W"%i)
                    _fcb = tf.Variable(tf.constant(value=0.1, dtype=np.float32, shape=[self.num_hidden[i]]),
                                     name="fc%d_b"%i)
                else:
                    _fcW = tf.Variable(tf.truncated_normal(shape=[self.num_hidden[i-1], self.num_hidden[i]],mean=0.0, stddev=0.1,dtype=np.float32),
                                  name="fc%d_W"%i)
                    _fcb = tf.Variable(tf.constant(value=0.1, dtype=np.float32, shape=[self.num_hidden[i]]),
                                     name="fc%d_b"%i)
                fcW.append(_fcW)
                fcb.append(_fcb)
                
                if i==len(self.num_hidden)-1:
                    _fcW = tf.Variable(tf.truncated_normal(shape=[self.num_hidden[i], num_labels],mean=0.0, stddev=0.1,dtype=np.float32),
                                  name="fc%d_W"%i)
                    _fcb = tf.Variable(tf.constant(value=0.1, dtype=np.float32, shape=[num_labels]),
                                     name="fc%d_b"%i)
                    fcW.append(_fcW)
                    fcb.append(_fcb)
                
                
            # 2. 定义图谱中的运算
            def model(data):
                print("create model")  
                # 被打印了两次，说明是在建立图谱时调用的()，而不是在训练/测试过程中反复调用
                shape = data.get_shape().as_list()
                reshaped = tf.reshape(data, [shape[0], shape[1]*shape[2]*shape[3]])
#                 hidden = tf.nn.relu(tf.matmul(reshaped, fc1_W)+fc1_b)
#                 output = tf.matmul(hidden, fc2_W) + fc2_b
                hidden = tf.nn.relu(tf.matmul(reshaped, fcW[0])+fcb[0])

                for i in range(1,len(fcW)-1):
                    hidden = tf.nn.relu(tf.matmul(hidden, fcW[i])+fcb[i])
                output = tf.matmul(hidden, fcW[-1]) + fcb[-1]
                return output

            # Training computation.
            logits = model(self.tf_trX) # 第一次建立模型图谱
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits, self.tf_trY))
            
            # Optimizer
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(self.loss)
            
            # Predictions for the training, validation, and test data.
            self.train_prediction = tf.nn.softmax(logits)
            self.test_prediction = tf.nn.softmax(model(self.tf_tsX))    # 第二次建立模型的图谱
            
            
    def run(self):
        """
        使用 session
        """
        self.sess = tf.Session(graph=self.graph)
        with self.sess as mySess:
            tf.global_variables_initializer().run()
            
            print("starting training")
            for i, samples, labels in get_chunk(TR_SAMPLES, TR_LABELS, chunkSize=self.batch_size):
                # 一次执行多个计算图节点
                _, l, predictions = mySess.run(
                        [self.optimizer, self.loss, self.train_prediction],
                        feed_dict={self.tf_trX: samples, self.tf_trY: labels}
                    )
                accuracy, _ = self.accuracy(predictions, labels)
                if i % 50 == 0:
                    print('Minibatch loss at step %d: %f' % (i, l))
                    print('Minibatch accuracy: %.1f%%' % accuracy)
       
            ### 测试
            accuracies = []
            confusionMatrices = []
            for i, samples, labels in get_chunk(TS_SAMPLES, TS_LABELS, chunkSize=self.test_batch_size):
                result = self.test_prediction.eval(feed_dict={self.tf_tsX: samples})
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
    net = Network(num_hidden=[128,64], batch_size=1000)
    net.define_graph()
    net.run()

    