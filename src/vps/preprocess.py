#-*-coding:UTF-8-*-
'''
Created on 2017 - 9 - 27 - 4:33:50 p.m.
author: Gary-W
'''
import os
import tensorflow as tf
from tf_dataio.tfrecords_io import ExampleReader
import numpy as np
import cv2
from util.image_augmentation import ImageTransformer

class PoseYCL:
    """ Pose Data预处理(YCL) 模块
    对于单个图像样本进行预处理，并提供预处理函数给tfrecords_io进行数据封装
    """
    def __init__(self):
        pass
    
    contraststretch = 0.2
    resize = (224, 224)
    display_augementation = False
    augment_size = 1
    
    @staticmethod
    def _make_label_arr_(labelstr):
        pass
        label_arr = np.array([float(num) for num in labelstr.split(" ")])
        label_xy = label_arr[:2]
        label_ab = label_arr[2:]
        return label_xy, label_ab
    
    @staticmethod
    def _preprocess_image_(attrs):
        """ 写入tfrecord文件前的预处理 """
        img_path = os.path.join(PoseYCL.raw_data_path, PoseYCL.data_type, attrs["name"])
        image = cv2.imread(img_path)

        # 裁剪中心方形ROI
        image = ImageTransformer.centered_crop(image, 224, 224)

        # 随机进行强度拉伸
#         image = ImageTransformer.random_contrast_stretching(image, PoseYCL.contraststretch)
        
        # 缩放到目标大小
        image = cv2.resize(image,PoseYCL.resize)
        
        if PoseYCL.display_augementation:
            cv2.imshow("augementation", image)
            cv2.waitKey(10)

        # 将np.array 转换为一个字符串
        image_byte = image.tobytes()
        return image_byte


def process_log(str_log):
    """ tfrecord输入日志解析数据进行封装的demo
        以OCR作为例子 对读取的图像进行数据争强后返回feature 列表
    """
    data_attrs = {}
    items = str_log.strip("\n").split(",")
    data_attrs["path"] = items[0]
    label_xy, label_ab = PoseYCL._make_label_arr_(items[1])
    data_attrs["pose_xy"] = label_xy
    data_attrs["pose_ab"] = label_ab
    
    feature_list = []
    for _i in range(PoseYCL.augment_size):
        try:
            features = {}
            # 转换为tfrecord的各项属性
            features['image'] = ExampleReader._bytes_feature(PoseYCL._preprocess_image_(data_attrs))
            features['pose_xy'] = ExampleReader._float_List_feature(data_attrs["label"])
            features['pose_ab'] = ExampleReader._float_List_feature(data_attrs["label"])
            
            feature_list.append(features)
        except:
            print("get invalid preprocess skipped of %s" % data_attrs["name"])
            
    return feature_list



if __name__=="__main__":
    pass

