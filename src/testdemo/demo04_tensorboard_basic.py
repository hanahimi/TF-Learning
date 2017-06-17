#coding:UTF-8
'''
Created on 2017年1月30日
@author: Ayumi Phoenix

演示tensorboard的基本使用方法
在board的父目录下，运行cmd
tensorboard --logdir board
http://192.168.1.105:6006/
使用chrome浏览器，使用360浏览器（极速模式）无法查看迭代曲线

目标(新增的操作)：
1. 定义计算图，并根据需求将目标变量加入summary用于TB的显示
    1.1 summary可以存储histogram, scalar, image
        histogram: 存放网络权重参数W,b之类的矩阵数据
        scalar：存放loss值之类的单值数据
        image：存放特征图像,卷积层结果（本例未用到...）
        
    1.2 为了TB绘制出整洁的图像，可以用tf.name_scope对变量进行打包
        TF中所有的计算都是计算图中的OP，因此打包的作用域是任意的
        可以将变量，优化，损失，甚至计算式等分别打包起来
    
    1.3 计算图中的变量和OP最好给个名字,否则显示时不知道谁是谁
    
    1.4 在计算图的最后使用merge_all将要显示的计算节点进行合并
        self.merged = tf.summary.merge_all()
    
2. 在计算图之外定义FileWriter,并绑定目标计算图
    eg. self.writer = tf.summary.FileWriter('./board', self.myGraph)

3. 启动会话
    3.1 在每一次迭代中，同时执行merge
        ..., mySummary = mySess.run(..., self.merge, feed_dict={...})
    3.2 将迭代得到的summary 添加到FileWriter中
        self.writer.add_summary(mySummary, i)
        
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

        # 统计
        self.merged = None 
        self.define_graph()
        self.sess = tf.Session(graph=self.graph)
        self.writer = tf.summary.FileWriter('./board', self.graph)  # 'tf.train.FileWriter' is depecated

        
    def define_graph(self):
        """ 定义计算图谱 """
        with self.graph.as_default():
            # 1. 定义图谱中的变量
            with tf.name_scope('inputs'):   # 对变量进行打包，使得tensorboard的图像工整一些
                self.tf_trX = tf.placeholder(dtype=np.float32, shape=(self.batch_size, image_size,image_size,num_channels), name='trX')
                self.tf_trY = tf.placeholder(dtype=np.float32, shape=(self.batch_size, num_labels), name='trY')
                self.tf_tsX = tf.placeholder(dtype=np.float32, shape=(self.test_batch_size, image_size,image_size,num_channels), name='tsX')
            
            with tf.name_scope('fc1'):
                fc1_W = tf.Variable(tf.truncated_normal(shape=[image_size**2, self.num_hidden],
                                                        mean=0.0, 
                                                        stddev=0.1,
                                                        dtype=np.float32),
                                          name="fc1_W")
                fc1_b = tf.Variable(tf.constant(value=0.1,
                                                dtype=np.float32,
                                                shape=[self.num_hidden]),
                                         name="fc1_b")
                # 添加进行迭代观察的变量
                tf.summary.histogram('fc1_W', fc1_W)
                tf.summary.histogram('fc1_b', fc1_b)

            with tf.name_scope('fc2'):
                fc2_W = tf.Variable(tf.truncated_normal(shape=[self.num_hidden, num_labels],
                                                        mean=0.0, 
                                                        stddev=0.1,
                                                        dtype=np.float32),
                                          name="fc2_W")
                fc2_b = tf.Variable(tf.constant(value=0.1, 
                                                dtype=np.float32, 
                                                shape=[num_labels]),
                                         name="fc2_b")
                tf.summary.histogram('fc2_W', fc2_W)
                tf.summary.histogram('fc2_b', fc2_b)
            
            # 2. 定义图谱中的运算
            def model(data):
                shape = data.get_shape().as_list()
                reshaped = tf.reshape(data, [shape[0], shape[1]*shape[2]*shape[3]])

                with tf.name_scope('fc1_model'):
                    fc1_model = tf.matmul(reshaped, fc1_W)+fc1_b
                    hidden = tf.nn.relu(fc1_model)
                    
                with tf.name_scope('fc2_model'):
                    output = tf.matmul(hidden, fc2_W) + fc2_b

                return output

            # Training computation.
            logits = model(self.tf_trX) # 第一次建立模型图谱
            
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits, self.tf_trY))
                tf.summary.scalar('loss', self.loss)
                
            with tf.name_scope('optimizer'):
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(self.loss)
            
            # Predictions for the training, validation, and test data.
            with tf.name_scope('prediction'):
                self.train_prediction = tf.nn.softmax(logits, name='train_prediction')
                self.test_prediction = tf.nn.softmax(model(self.tf_tsX), name='test_prediction')    # 第二次建立模型的图谱

            self.merged = tf.summary.merge_all()

            
    def run(self):
        """ 启动会话，执行计算图 """
        with self.sess as mySess:
            # 只有初始化过所有变量以后，summarywriter 才能知道网络的信息
            tf.global_variables_initializer().run()
            for i, samples, labels in get_chunk(TR_SAMPLES, TR_LABELS, chunkSize=self.batch_size):
                # 一次执行多个计算图节点
                _, l, predictions,summary = mySess.run(
                        [self.optimizer, self.loss, self.train_prediction, self.merged],
                        feed_dict={self.tf_trX: samples, self.tf_trY: labels}
                    )
                self.writer.add_summary(summary, i)
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
    net = Network(num_hidden=128, batch_size=1000)
    net.run()

