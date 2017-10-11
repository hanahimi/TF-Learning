#-*-coding:UTF-8-*-
'''
Created on 2017年9月20日-下午5:39:30
author: Gary-W
ref:
 TFRecords 文件的生成和读取 http://blog.csdn.net/u012222949/article/details/72875281
 tfrecords文件的制作工具 http://blog.csdn.net/m0_37041325/article/details/74891322

TensorFlow提供了TFRecords的格式来统一存储数据，理论上，TFRecords可以存储任何形式的数据
TFRecords文件中的数据都是通过tf.train.Example Protocol Buffer的格式存储的
以下的代码给出了tf.train.Example的定义
message Example {  
    Features features = 1;  
};  
message Features {  
    map<string, Feature> feature = 1;  
};  
message Feature {  
    oneof kind {  
    BytesList bytes_list = 1;  
    FloatList float_list = 2;  
    Int64List int64_list = 3;  
}  
}; 
因此tfrecords的关键函数就是将要保存的数据转换为上述对应的格式

 创建一个属性（feature）用于保存  
feature = {'train/label': _int64_feature(label),  
           'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}  
# 读取时
feature = {  
    'train/image': tf.FixedLenFeature([], tf.string),  
    'train/label': tf.FixedLenFeature([], tf.int64)  
}

'''
import os
import random
import shutil
import tensorflow as tf

class ExampleReader(object):
    def __init__(self):
        pass
    
    # 将数据转化成对应的属性
    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))  
        
    @staticmethod
    def _int64_List_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
        
    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))  

    @staticmethod
    def _float_feature(value):  
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))  
    
    @staticmethod
    def _float_List_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    
    
    @staticmethod
    def convert_logs_to_tfrecords(src_dataset_log_path, dst_tf_records_path, processfunc):
        """ 
        将目标log中的数据封装成tfrecords文件
        输入：
            src_dataset_log_path：源数据集列表文本路径
            dst_tf_records_path：目标tf_records文件路径
            processfunc： 用户提供的处理函数 feature = processfunc(strlog)
    
        用户需要实现 processfunc 函数，将logs中的数据转换为tfcords的feature字典(or 字典的列表)
        将logs中的数据转换的tf example 并写入 到tfrecords
        tf 按顺序写入，在测试时随机batch 或者给定的log就是shuffle过的
        """
        # 解析数据文本文件
        dataset_log_path = src_dataset_log_path
        dataset_log_list = []
        with open(dataset_log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    dataset_log_list.append(line)
        data_num = len(dataset_log_list)
        print("data_num = %d\n" % data_num)
        
        # 检查是否已经存在对应的数据文件，已有则删除
        tf_records_path = dst_tf_records_path
        if os.path.exists(tf_records_path):
            os.remove(tf_records_path)
        
        writer = tf.python_io.TFRecordWriter(tf_records_path)
        get_data_num = 0
        for i in range(data_num):
            if not i % 10:
                print("converting: %d/%d %s" % (i, data_num, dataset_log_list[i]))
    
            # 创建对应的属性列表（兼容由一个样本增强为多个样本的情况）
            featurelist = processfunc(dataset_log_list[i])
            
            # 检查是否为多个样本
            if type(featurelist) == list:
                for fi in range(len(featurelist)):
                    feature = featurelist[fi]
                    # 创建一个example protocol buffer 
                    example = tf.train.Example(features=tf.train.Features(feature=feature))  
                    writer.write(example.SerializeToString())
                    get_data_num += 1
    
            # 检查是否为单个样本
            elif type(featurelist) == dict:
                feature = featurelist
                example = tf.train.Example(features=tf.train.Features(feature=feature))  
                writer.write(example.SerializeToString())
                get_data_num += 1
            else:
                print("unknow feature")
            
        writer.close()
        print("tfrecord convert finish data numbers: %d\n" % get_data_num)
        return get_data_num

    @staticmethod
    def touch_tfrecords(src_tf_records_path, processfunc, feature):
        """ 
        从tfrecords文件中还原数据
        输入：
            src_tf_records_path：源tf_records文件路径
            processfunc： 用户提供的处理函数 processfunc(feature)
            feature: 定义feature，这里要和之前创建的时候保持一致
            eg.
            feature = {  
                    'train/image': tf.FixedLenFeature([], tf.string),  
                    'train/label': tf.FixedLenFeature([], tf.int64)  
                }
        用户需要实现 processfunc函数，对feature进行解析以及后处理
        """
        
        # 创建一个队列来维护输入文件列表  
        filename_queue = tf.train.string_input_producer([src_tf_records_path], num_epochs=1)  
        
        reader = tf.TFRecordReader()     # 定义一个 reader ，读取下一个 record  
        _, serialized_example = reader.read(filename_queue) 
        
        # 解析读入的一个record  
        features = tf.parse_single_example(serialized_example, features=feature) 
        
        # 对feature见解析
        data_batch = processfunc(features)
     
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            
            
            coord.request_stop()
            coord.join(threads)
            print('Finished')
        
if __name__=="__main__":
    pass

