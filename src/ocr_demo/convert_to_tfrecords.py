'''
Created on 2017年9月22日-下午2:25:34
author: Gary-W

'''
import os
import tensorflow as tf
from tf_dataio.tfrecords_io import ExampleReader
from util.image_augmentation import ImageTransformer
import numpy as np
import cv2


class OCRYCL:
    """ OCR 预处理(YCL) 模块 (主要是preprocess太长了:| ..)
    用于控制不同数据集中的数据增强方式，并提供预处理函数给tfrecords_io进行数据封装
    """
    def __init__(self):
        pass
    label_table = {"0":0,
                        "1":1, "2":2, "3":3,
                        "4":4, "5":5, "6":6,
                        "7":7, "7":8, "7":9,
                        "A":10, "B":11, 
                        "E":12, "F":13,
                        "*":14, "Nan":15}
    max_ocr_length = 3
    num_negtive = label_table["Nan"]
        
    raw_data_path = r"your_raw_dataset_dir_path"
    data_type = "train"

    augment_size = 20       # 一个样本最多增幅的次数
    crop_scale = 1.3        # 
    contraststretch = 0.5
    exchannel = 0.3
    rotate = 10
    resize = (64,64)
    display_preprocess = False

    @staticmethod
    def _set_preprocess(
                        data_type = "train",
                        augment_size = 20,
                        crop_scale = 1.3,
                        contraststretch=0.5,
                        exchannel = 0.3,
                        rotate = 10,
                        resize = (64,64),
                        display_augementation = False
                        ):
        OCRYCL.data_type = data_type
        OCRYCL.augment_size = augment_size
        OCRYCL.crop_scale = crop_scale
        OCRYCL.contraststretch = contraststretch
        OCRYCL.exchannel = exchannel
        OCRYCL.rotate = rotate
        OCRYCL.resize = resize
        OCRYCL.display_augementation = display_augementation

        
    @staticmethod
    def _make_label_arr_(labelstr):
        # 将字符串进行编码 为 固定长度的
        label_arr = np.zeros(OCRYCL.max_ocr_length, np.int)
        for idx, _chr in enumerate(labelstr):
            label_arr[idx] = int(OCRYCL.label_table[_chr] \
                                 if _chr in OCRYCL.label_table else OCRYCL.num_negtive)
        return label_arr
    
    @staticmethod
    def _preprocess_image_(attrs):
        """ 写入tfrecord文件前的预处理 """
        img_path = os.path.join(OCRYCL.raw_data_path, OCRYCL.data_type, attrs["name"])
        image = cv2.imread(img_path)

        # 围绕ROI中心随机裁剪方形区域
        scale = 1.3
        center_x = int(round((attrs["right"] + attrs["left"]) / 2.0))
        center_y = int(round((attrs["top"] + attrs["bottom"]) / 2.0))
        max_side = max(attrs["right"] - attrs["left"], attrs["bottom"] - attrs["top"])
        bbox_left, bbox_top, bbox_width, bbox_height = (center_x - int(max_side/2.0), center_y - int(max_side/2.0), max_side, max_side)
        image = ImageTransformer.random_crop(image, bbox_left, bbox_top, bbox_width, bbox_height, scale)

        # 随机进行强度拉伸
        image = ImageTransformer.random_contrast_stretching(image, OCRYCL.contraststretch)

        # 随机交换通道
        image = ImageTransformer.random_exchange_channel(image, OCRYCL.exchannel)

        # 随机旋转
        image = ImageTransformer.random_rotation(image, OCRYCL.rotate)
        
        # 缩放到目标大小
        image = cv2.resize(image,OCRYCL.resize)
        
        # 归一化
        
        if OCRYCL.display_augementation:
            cv2.imshow("augementation", image)
            cv2.waitKey(10)

        # 将np.array 转换为一个字符串
        image_byte = image.tobytes()
        return image_byte

    @staticmethod
    def preprocess_reading(image):
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.multiply(tf.subtract(image, 0.5), 2) # 这一步的原因是什么。。。是否减去了均值
        image = tf.reshape(image, [64, 64, 3])
        return image


def process_log(str_log):
    """ tfrecord输入日志解析数据进行封装的demo
        以OCR作为例子 对读取的图像进行数据争强后返回feature 列表
    """
    data_attrs = {}
    items = str_log.split(" ")
    data_attrs["name"] = items[0]
    data_attrs["label"] = OCRYCL._make_label_arr_(items[1])
    data_attrs["length"] = len(items[1])
    data_attrs["left"] = int(items[2])
    data_attrs["top"] = int(items[3])
    data_attrs["right"] = int(items[4])
    data_attrs["bottom"] = int(items[5])
    feature_list = []
    for _i in range(OCRYCL.augment_size):
        try:
            features = {}
            # 转换为tfrecord的各项属性
            features['image'] = ExampleReader._bytes_feature(OCRYCL._preprocess_image_(data_attrs))
            features['length'] = ExampleReader._int64_feature(data_attrs["length"])
            features['label'] = ExampleReader._int64_List_feature(data_attrs["label"])
            features['name'] = ExampleReader._bytes_feature(str.encode(data_attrs["name"]))
            feature_list.append(features)
        except:
            print("get invalid preprocess skipped of %s" % data_attrs["name"])
            
    return feature_list
        
    
def convert_ocr_to_tfrecords(src_dataset_log_path, dst_tf_records_path):
    processfunc = process_log
    sample_num = ExampleReader.convert_logs_to_tfrecords(src_dataset_log_path, dst_tf_records_path, processfunc)
    return sample_num


def decode_tfrecords_to_ocr(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'length': tf.FixedLenFeature([], tf.int64),
                'label': tf.FixedLenFeature([OCRYCL.max_ocr_length], tf.int64)
            })
    image = Donkey._preprocess(tf.decode_raw(features['image'], tf.uint8))
    length = tf.cast(features['length'], tf.int32)
    digits = tf.cast(features['label'], tf.int32)
    return image, length, digits

    

def main(sel=1):
    raw_dataset = r"E:\OCR-PLD_Dataset\saic_dataset\raw_data_100x100"
    dst_dataset = r"E:\OCR-PLD_Dataset\saic_dataset\tfdataset"
    dataset_name = "0924"
    
    if sel==1:
        OCRYCL.raw_data_path = raw_dataset
        
        # make training set
        train_dataset_log_path = os.path.join(dst_dataset, dataset_name, "train_dataset.txt")
        train_tfrecords_path = os.path.join(dst_dataset, dataset_name, "train.tfrecords")
        OCRYCL._set_preprocess(data_type="train",crop_scale=1.3, contraststretch=0.5, exchannel=0.3, rotate=10, resize=(64,64))
        train_num = convert_ocr_to_tfrecords(train_dataset_log_path, train_tfrecords_path)
    
        # make validing set
        val_dataset_log_path = os.path.join(dst_dataset, dataset_name, "val_dataset.txt")
        val_tfrecords_path = os.path.join(dst_dataset, dataset_name, "val.tfrecords")
        OCRYCL._set_preprocess(data_type="val",crop_scale=1.3, contraststretch=0.3, exchannel=0, rotate=2, resize=(64,64))
        val_num = convert_ocr_to_tfrecords(val_dataset_log_path, val_tfrecords_path)
        
        # make testing set
        test_dataset_log_path = os.path.join(dst_dataset, dataset_name, "test_dataset.txt")
        test_tfrecords_path = os.path.join(dst_dataset, dataset_name, "test.tfrecords")
        OCRYCL._set_preprocess(data_type="test",crop_scale=1.3, contraststretch=0, exchannel=0, rotate=2, resize=(64,64))
        test_num = convert_ocr_to_tfrecords(test_dataset_log_path, test_tfrecords_path)
        
        # make meta.json file
        meta_msg = '{"num_examples": {"train": %d, "val": %d, "test": %d}}' % (train_num, val_num, test_num)
        print(meta_msg)
        with open(os.path.join(dst_dataset, dataset_name, "meta.json"), "w") as mf:
            mf.write(meta_msg)

    elif sel == 2:
        # 从tfrecords中还原数据
        test_dataset_log_path = os.path.join(dst_dataset, dataset_name, "test_dataset.txt")
        read_tfrecords_to_ocr(test_dataset_log_path)
        
if __name__=="__main__":
    pass
    main(1)
    







