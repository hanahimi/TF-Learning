# Import the converted model's class
import numpy as np
import random
import tensorflow as tf
from poselstm.posenet import GoogLeNet as PoseNet
import cv2
import os
from tqdm import tqdm

BATCH_SIZE = 32
max_iterations = 30000
# Set this path to your dataset directory
directory = r'D:\loc_train'
dataset = 'dataset_train.txt'

class datasource(object):
	def __init__(self, images, poses):
		self.images = images
		self.poses = poses

def centeredCrop(img, output_side_length):
	height, width, depth = img.shape
	new_height = output_side_length
	new_width = output_side_length
	if height > width:
		new_height = output_side_length * height / width
	else:
		new_width = output_side_length * width / height
	height_offset = int((new_height - output_side_length) / 2)
	width_offset = int((new_width - output_side_length) / 2)
	cropped_img = img[height_offset:height_offset + output_side_length,
						width_offset:width_offset + output_side_length]
	return cropped_img

def preprocess(images):
	images_out = [] #final result
	#Resize and crop and compute mean!
	images_cropped = []
	for i in tqdm(range(len(images))):
		X = cv2.imread(images[i])
		X = cv2.resize(X, (455, 256))
		X = centeredCrop(X, 224)
		images_cropped.append(X)
	#compute images mean
	N = 0
	mean = np.zeros((1, 3, 224, 224))
	for X in tqdm(images_cropped):
		mean[0][0] += X[:,:,0]
		mean[0][1] += X[:,:,1]
		mean[0][2] += X[:,:,2]
		N += 1
	mean[0] /= N
	#Subtract mean from all images
	for X in tqdm(images_cropped):
		X = np.transpose(X,(2,0,1))
		X = X - mean
		X = np.squeeze(X)
		X = np.transpose(X, (1,2,0))
		images_out.append(X)
	return images_out

def get_data():
	poses = []
	images = []
	with open(os.path.join(directory,dataset)) as f:
		next(f)  # skip the 3 header lines
		next(f)
		next(f)
		for line in f:
			fname, p0,p1,p2,p3 = line.split()
			p0 = float(p0)
			p1 = float(p1)
			p2 = float(p2)
			p3 = float(p3)
			poses.append((p0,p1,p2,p3))
			images.append(os.path.join(directory,fname))
			if len(images) > 100:
				break
	images = preprocess(images)
	return datasource(images, poses)

def gen_data(source):
	while True:
		indices = list(range(len(source.images)))
		random.shuffle(indices)
		for i in indices:
			image = source.images[i]
			pose_x = source.poses[i][0:2]
			pose_q = source.poses[i][2:4]
			yield image, pose_x, pose_q

def gen_data_batch(source):
	data_gen = gen_data(source)
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


def main():
	images = tf.placeholder(tf.float32, [BATCH_SIZE, 224, 224, 3])
	poses_x = tf.placeholder(tf.float32, [BATCH_SIZE, 2])
	poses_q = tf.placeholder(tf.float32, [BATCH_SIZE, 2])
	datasource = get_data()

	net = PoseNet({'data': images})

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


	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	outputFile = r"D:\loc_train\PoseNet.ckpt"

	with tf.Session() as sess:
		# Load the data
		sess.run(init)
		net.load(r'D:\loc_train\posenet.npy', sess)

		data_gen = gen_data_batch(datasource)
		for i in range(max_iterations):
			np_images, np_poses_x, np_poses_q = next(data_gen)
			feed = {images: np_images, poses_x: np_poses_x, poses_q: np_poses_q}

			sess.run(opt, feed_dict=feed)
			np_loss = sess.run(loss, feed_dict=feed)
			if i % 10 == 0:
				print("iteration: " + str(i) + "\n\t" + "Loss is: " + str(np_loss))
			if i % 100 == 0:
				saver.save(sess, outputFile)
				print("Intermediate file saved at: " + outputFile)
		saver.save(sess, outputFile)
		print("Intermediate file saved at: " + outputFile)


if __name__ == '__main__':
	main()


