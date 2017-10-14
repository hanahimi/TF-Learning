#-*-coding:UTF-8-*-
'''
Created on 2017 - 10 - 14 - 9:13:16 am
author: Gary-W
'''
import numpy as np
import tensorflow as tf
from vps.posenet import Posenet
import random
import cv2
import os
from tqdm import tqdm
from util.image_augmentation import ImageTransformer


BATCH_SIZE = 1
# Set this path to your dataset directory
directory = r'D:\Ayumi\workspace\dataset\vps_dataset'

dataname = r'dataset_0925S5'
dataset_npy = dataname+"_train.npy"
save_ckpt = dataname+".ckpt"

DS_PATH = os.path.join(directory,dataname,"dataset", dataset_npy)
FINETUNE_WEIGHTS_PATH = os.path.join(directory,dataname,"models","places_googlenet.npy")
OUTPUT_FILE = os.path.join(directory,dataname,"save",save_ckpt)

class DataSource(object):
    def __init__(self):
        self.images = []
        self.poses = []
        self.mean_img = None
    
    def set_source(self, images, poses, mean_img):
        self.images = images
        self.poses = poses
        self.mean_img = mean_img
        
    def gen_data(self):
        while True:
            indices = list(range(len(self.images)))
            random.shuffle(indices)
            for i in indices:
                image = self.images[i]
                pose_x = self.poses[i][0:2]
                pose_q = self.poses[i][2:4]
                yield image, pose_x, pose_q

    def gen_data_batch(self, BATCH_SIZE = 32):
        data_gen = self.gen_data()
        while True:
            image_batch = []
            pose_x_batch = []
            pose_q_batch = []
            for _ in range(BATCH_SIZE):
                image, pose_x, pose_q = next(data_gen)
                image_batch.append(image)
                pose_x_batch.append(pose_x)
                pose_q_batch.append(pose_q)
            yield np.array(image_batch), np.array(pose_x_batch), np.array(pose_q_batch)

def get_dataset():
    print("loading dataset...")
    dict_data = np.load(DS_PATH, encoding="latin1").item()
    ds = dict_data["train"]
    print("size: ", len(ds.images))
    print("data shape: ", ds.images[0].shape, " max:",np.max(ds.images[0]), " min:",np.min(ds.images[0]))
    print("label shape: ", ds.poses[0].shape)
    return ds

        
def main():
    datasource = get_dataset()
    
    images = tf.placeholder(tf.float32, [1, 224, 224, 3])
    results = np.zeros((len(datasource.images),2))
    
    net = Posenet(images)
    
    p3_x = net.layers['cls3_fc_pose_xy']
    p3_q = net.layers['cls3_fc_pose_ab']
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Load the net
        sess.run(init)
        saver.restore(sess, OUTPUT_FILE)
        # Load the data
        
        for i in range(len(datasource.images)):
            np_image = datasource.images[i]
            np_image = np.expand_dims(np_image, 0)
            feed = {images: np_image}
            
            pose_q= np.asarray(datasource.poses[i][2:4])
            pose_x= np.asarray(datasource.poses[i][0:2])
            predicted_x, predicted_q = sess.run([p3_x, p3_q], feed_dict=feed)

            pose_q = np.squeeze(pose_q)
            pose_x = np.squeeze(pose_x)
            predicted_q = np.squeeze(predicted_q)
            predicted_x = np.squeeze(predicted_x)
            
            #Compute Individual Sample Error0
            q1 = pose_q / np.linalg.norm(pose_q)
            q2 = predicted_q / np.linalg.norm(predicted_q)
            d = abs(np.sum(np.multiply(q1,q2)))
            theta = 2 * np.arccos(d) * 180/np.pi
            error_x = np.linalg.norm(pose_x-predicted_x)
            results[i,:] = [error_x,theta]
            print('Iteration:  ', i, '  Error XYZ (m):  ', error_x, '  Error Q (degrees):  ', theta)
    median_result = np.median(results,axis=0)
    print('Median error ', median_result[0], 'm  and ', median_result[1], 'degrees.')

if __name__=="__main__":
    pass
    main()

