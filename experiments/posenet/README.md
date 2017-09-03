# TensorFlow-PoseNet
**This is an implementation for Tensorflow of [PoseNet architecture](http://mi.eng.cam.ac.uk/projects/relocalisation/) with tensorflow, modified to run in python3.5-windows**
The way to construct the network is applying [caffe-windows tools]()

Reference Paper:
[1] ICCV 2015 paper **PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization** Alex Kendall, Matthew Grimes and Roberto Cipolla [http://mi.eng.cam.ac.uk/projects/relocalisation/]

## Getting Started
 * Download the Cambridge Landmarks King's College dataset [from here.](https://www.repository.cam.ac.uk/handle/1810/251342)
 * Download the starting and trained weights [from here.](https://drive.google.com/file/d/0B5DVPd_zGgc8ZmJ0VmNiTXBGUkU/view?usp=sharing)
	I have upload them to  baiduyun: s/1o7Rhusi fftu"
 * The PoseNet model is defined in the posenet.py file

 * The starting and trained weights (posenet.npy and PoseNet.ckpt respectively) for training were obtained by converting caffemodel weights [from here](http://vision.princeton.edu/pvt/GoogLeNet/Places/) and then training.

 * To run:
   * Extract the King's College dataset to wherever you prefer
   * Extract the starting and trained weights to wherever you prefer
   * Update the datasetpaths on  train.py as well as test.py
   * If you want to retrain, simply run train.py (note this will take a long time)
   * If you just want to test, simply run test.py 