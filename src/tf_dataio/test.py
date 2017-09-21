#-*-coding:UTF-8-*-
'''
Created on 2017年9月20日-下午8:18:58
author: Gary-W
'''
from tf_dataio.tfrecords_io import ExampleReader

def process_example(str_log):
    """ tfrecord输入日志解析数据进行封装的demo
    以OCR作为例子
    """
    items = str_log.split(" ")
    img_path = items[0]
    label = items[1]


if __name__=="__main__":
    pass

