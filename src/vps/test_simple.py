#-*-coding:UTF-8-*-
'''
Created on 2017Äê10ÔÂ12ÈÕ
@author: mipapapa
'''

import numpy as np
import tensorflow as tf
from vps.posenet import Posenet
import random
import cv2
import os
from tqdm import tqdm
from util.image_augmentation import ImageTransformer

BATCH_SIZE = 75
max_iterations = 30000
# Set this path to your dataset directory
directory = r'D:\Env_WJR\dataset\tf_vps\0924S5_SEL'
dataset = 'dataset_0925S5'
tr_dataset = dataset+"_test.txt"
dataset_npy = dataset+"_test.npy"
OUTPUT_FILE = r"D:\Env_WJR\dataset\tf_vps\0924S5_SEL\dataset_0925S5\saver\dataset_0925S5.ckpt"

class PoseYCL:
    """ Pose Data preprocess(YCL) module
    preprocess an image
    """
    def __init__(self):
        pass
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
    def _preprocess_image_(img_path):
        image = cv2.imread(img_path)
        image = ImageTransformer.centered_crop(image, 224, 224)
        if image.shape[:2] != PoseYCL.resize:
            image = cv2.resize(image,PoseYCL.resize)
        return image

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

    def gen_data_batch(self, batch_size = 32):
        data_gen = self.gen_data()
        while True:
            image_batch = []
            pose_x_batch = []
            pose_q_batch = []
            for _ in range(batch_size):
                image, pose_x, pose_q = next(data_gen)
                image_batch.append(image)
                pose_x_batch.append(pose_x)
                pose_q_batch.append(pose_q)
            yield np.array(image_batch), np.array(pose_x_batch), np.array(pose_q_batch)



def preprocess(img_path_list):
    images_out = [] #final result

    images_cropped = []
    for i in tqdm(range(len(img_path_list))):
        X = PoseYCL._preprocess_image_(img_path_list[i])
        images_cropped.append(X)
    
    # compute images mean
    N = 0
    img_mean = np.zeros((224,224,3), np.float64)
    for X in tqdm(images_cropped):
        img_mean[:] += X[:]
        N += 1
    img_mean /= N

    #Subtract mean from all images
    for X in tqdm(images_cropped):
        X = X - img_mean    # X become float64, donot use: X[:]=X[:]-img_mean[:])
        assert X.shape == (224,224,3)
        images_out.append(X)
    return images_out, img_mean

def get_dataset():
    ds_path = os.path.join(directory,dataset,dataset_npy)
    if os.path.exists(ds_path):
        print("loading dataset...")
        dict_data = np.load(ds_path, encoding="latin1").item()
        ds = dict_data["train"]
        print("size: ", len(ds.images))
        print("data shape: ", ds.images[0].shape, " max:",np.max(ds.images[0]), " min:",np.min(ds.images[0]))
        print("label shape: ", ds.poses[0].shape)
        
        
    else:        
        poses = []
        images = []
        with open(os.path.join(directory,dataset,tr_dataset)) as f:
            for line in f:
                fpath, label = line.strip().split(",")
                pose_xyab = np.array([float(item) for item in label.split(" ")])
                poses.append(pose_xyab)
                images.append(fpath)
        images, mean_img = preprocess(images)
        ds = DataSource()
        ds.set_source(images, poses, mean_img)
        dict_data = {"train": ds}
        print("saving dataset...")
        np.save(ds_path, dict_data)
    
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
        _writer = tf.summary.FileWriter("logs/", sess.graph)
        sess.run(init)
        print("restoring:", OUTPUT_FILE)
        saver.restore(sess, OUTPUT_FILE)

        # Load the data
        for i in range(len(datasource.images)):
            np_image = datasource.images[i]
            # reshape to (1,244, 244, 3)
            np_image = np.expand_dims(np_image, 0)
            feed = {images: np_image}
            pose_x= np.asarray(datasource.poses[i][0:2])
            pose_q= np.asarray(datasource.poses[i][2:4])
            predicted_x, predicted_q = sess.run([p3_x, p3_q], feed_dict=feed)
            
            pose_q = np.squeeze(pose_q)
            pose_x = np.squeeze(pose_x)
            predicted_q = np.squeeze(predicted_q)
            predicted_x = np.squeeze(predicted_x)

            #Compute Individual Sample Error0
            q1 = pose_q / np.linalg.norm(pose_q)
            q2 = predicted_q / np.linalg.norm(predicted_q)
            d = abs(np.sum(np.multiply(q1,q2)))
            if d > 1: d = 2 - d
            theta = 2 * np.arccos(d) * 180/np.pi
            error_x = np.linalg.norm(pose_x-predicted_x)
            results[i,:] = [error_x,theta]
            print('Iteration:  ', i, '  Error XY (m):  ', error_x, '  Error Q (degrees):  ', theta)

    median_result = np.median(results,axis=0)
    print('Median error ', median_result[0], 'm  and ', median_result[1], 'degrees.')
        
if __name__=="__main__":
    pass
    main()

