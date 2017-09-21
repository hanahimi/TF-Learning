import os 
import tensorflow as tf 
import cv2
import numpy as np



def save_npy2tfrecords():
    cwd='D:\Python\data\dog\\'
    classes={'husky','chihuahua'} #��Ϊ �趨 2 ��
    writer= tf.python_io.TFRecordWriter("dog_train.tfrecords") #Ҫ���ɵ��ļ�
     
    for index,name in enumerate(classes):
        class_path=cwd+name+'\\'
        for img_name in os.listdir(class_path): 
            img_path=class_path+img_name #ÿһ��ͼƬ�ĵ�ַ
     
            img=Image.open(img_path)
            img= img.resize((128,128))
            img_raw=img.tobytes()#��ͼƬת��Ϊ�����Ƹ�ʽ
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            })) #example�����label��image���ݽ��з�װ
            writer.write(example.SerializeToString())  #���л�Ϊ�ַ���
     
    writer.close()
    



