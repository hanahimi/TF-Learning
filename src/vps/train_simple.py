#-*-coding:UTF-8-*-
'''
Created on 2017-10-10 - 3:32:24 p.m.
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

BATCH_SIZE = 32
max_iterations = 30000
# Set this path to your dataset directory
directory = r'D:\Env_WJR\dataset\tf_vps\0924S5_SEL'
dataset = 'dataset_0925S5'
tr_dataset = dataset+"_train.txt"
dataset_npy = dataset+".npy"
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

    images = tf.placeholder(tf.float32, [BATCH_SIZE, 224, 224, 3])
    poses_x = tf.placeholder(tf.float32, [BATCH_SIZE, 2])
    poses_q = tf.placeholder(tf.float32, [BATCH_SIZE, 2])

    weights_path = r"D:\Env_WJR\dataset\loc_train\googlenet\version_4_googlenet_places_short\places_googlenet.npy"
    net = Posenet(images, weights_path)
    skip_layer = ["cls1_reduction", "cls1_fc1", "cls1_fc2",
                  "cls2_reduction", "cls2_fc1", "cls2_fc2",
                  "cls3_fc"]
    
    p1_x = net.layers['cls1_fc_pose_xy']
    p1_q = net.layers['cls1_fc_pose_ab']
    p2_x = net.layers['cls2_fc_pose_xy']
    p2_q = net.layers['cls2_fc_pose_ab']
    p3_x = net.layers['cls3_fc_pose_xy']
    p3_q = net.layers['cls3_fc_pose_ab']

    
    l1_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p1_x, poses_x)))) * 0.3
    l1_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p1_q, poses_q)))) * 150
    l2_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p2_x, poses_x)))) * 0.3
    l2_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p2_q, poses_q)))) * 150
    l3_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p3_x, poses_x)))) * 1
    l3_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p3_q, poses_q)))) * 500

    loss = l1_x + l1_q + l2_x + l2_q + l3_x + l3_q
    opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=0.00000001, use_locking=False, name='Adam').minimize(loss)
#     sess_config = tf.ConfigProto()
#     sess_config.gpu_options.per_process_gpu_memory_fraction = 0.90
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        _writer = tf.summary.FileWriter("logs/", sess.graph)
#         sess.run(init)
        # Load the net
#         net.load_initial_weights(sess, skip_layer)

        print("restoring:", OUTPUT_FILE)
        saver.restore(sess, OUTPUT_FILE)

        # Load the data
        data_gen = datasource.gen_data_batch(32)
        for i in range(max_iterations):
            np_images, np_poses_x, np_poses_q = next(data_gen)
            feed = {images: np_images, poses_x: np_poses_x, poses_q: np_poses_q}
            sess.run(opt, feed_dict=feed)
            np_loss = sess.run(loss, feed_dict=feed)
            if i % 10 == 0:
                print("iteration: " + str(i) + "\n\t" + "Loss is: " + str(np_loss))
            if i % 100 == 0:
                saver.save(sess, OUTPUT_FILE)
                print("Intermediate file saved at: " + OUTPUT_FILE)
        saver.save(sess, OUTPUT_FILE)
        print("Intermediate file saved at: " + OUTPUT_FILE)
        
if __name__=="__main__":
    pass
    main()










