#-*-coding:UTF-8-*-
'''
Created on 2017-9-26-6:42:10 pm
author: Gary-W
'''
import tensorflow as tf
from vps.network_tookit import NW
import numpy as np

class Posenet(object):
    def __init__(self, x, weights_path="default"):
        """
        Inputs:
        x: placeholder for images
        weights_path: path string, path to pretraining weights, a npy file
        """
        self.X = x
        self.layers = {}
        self.WEIGHTS_PATH = weights_path
        self.SKIP_LAYER = []
        # call the create function to build the computational graph of posenet
        self.create()

    def create(self):
        """ create pose network (based on googlenet)"""
        
        """ A solo prepressing reduction network in the head """
        print("pre_reduction")
        with tf.name_scope('pre_reduction'):
            conv1 = NW.conv(self.X, 7, 7, 64, 2, 2, name='conv1')
            pool1 = NW.max_pool(conv1, 3, 3, 2, 2, name='pool1')
            norm1 = NW.lrn(pool1, 2, 2e-05, 0.75, name='norm1')
            reduction2 = NW.conv(norm1, 1, 1, 64, 1, 1, name='reduction2')
            conv2 = NW.conv(reduction2, 3, 3, 192, 1, 1,name='conv2')
            norm2 = NW.lrn(conv2, 2, 2e-05, 0.75, name='norm2')
            pool2 = NW.max_pool(norm2, 3, 3, 2, 2, name='pool2')
        
        """ 1st inception layer group """
        print("icp1")
        with tf.name_scope('icp1'):
            # branch 0
            icp1_out0 = NW.conv(pool2, 1, 1, 64, 1, 1, name='icp1_out0')
            # branch 1
            icp1_reduction1 = NW.conv(pool2, 1, 1, 96, 1, 1, name='icp1_reduction1')
            icp1_out1 = NW.conv(icp1_reduction1, 3, 3, 128, 1, 1, name='icp1_out1')
            # branch 2
            icp1_reduction2 = NW.conv(pool2, 1, 1, 16, 1, 1, name='icp1_reduction2')
            icp1_out2 = NW.conv(icp1_reduction2, 5, 5, 32, 1, 1, name='icp1_out2')
            # branch 3
            icp1_pool = NW.max_pool(pool2, 3, 3, 1, 1, name='icp1_pool')
            icp1_out3 = NW.conv(icp1_pool, 1, 1, 32, 1, 1, name='icp1_out3')
            # concat
            icp2_in = NW.concat([icp1_out0,
                                 icp1_out1,
                                 icp1_out2,
                                 icp1_out3], 3, 'icp2_in')

        """ 2nd inception layer group """
        print("icp2")
        with tf.name_scope('icp2'):
            # branch 0
            icp2_out0 = NW.conv(icp2_in, 1, 1, 128, 1, 1, name='icp2_out0')
            # branch 1
            icp2_reduction1 = NW.conv(icp2_in, 1, 1, 128, 1, 1, name='icp2_reduction1')
            icp2_out1 = NW.conv(icp2_reduction1, 3, 3, 192, 1, 1, name='icp2_out1')
            # branch 2
            icp2_reduction2 = NW.conv(icp2_in, 1, 1, 32, 1, 1, name='icp2_reduction2')
            icp2_out2 = NW.conv(icp2_reduction2, 5, 5, 96, 1, 1, name='icp2_out2')
            # branch 3
            icp2_pool = NW.max_pool(icp2_in, 3, 3, 1, 1, name='icp2_pool')
            icp2_out3 = NW.conv(icp2_pool, 1, 1, 64, 1, 1, name='icp2_out3')
            # concat
            icp2_out = NW.concat([icp2_out0,
                                 icp2_out1,
                                 icp2_out2,
                                 icp2_out3], 3, 'icp2_out')
        
        """ 3rd inception layer group """
        print("icp3")
        with tf.name_scope('icp3'):
            icp3_in = NW.max_pool(icp2_out, 3, 3, 2, 2, name='icp3_in')
            # branch 0
            icp3_out0 = NW.conv(icp3_in, 1, 1, 192, 1, 1, name='icp3_out0')
            # branch 1
            icp3_reduction1 = NW.conv(icp3_in, 1, 1, 96, 1, 1, name='icp3_reduction1')
            icp3_out1 = NW.conv(icp3_reduction1, 3, 3, 208, 1, 1, name='icp3_out1')
            # branch 2
            icp3_reduction2 = NW.conv(icp3_in, 1, 1, 16, 1, 1, name='icp3_reduction2')
            icp3_out2 = NW.conv(icp3_reduction2, 5, 5, 48, 1, 1, name='icp3_out2')
            # branch 3
            icp3_pool = NW.max_pool(icp3_in, 3, 3, 1, 1, name='icp3_pool')
            icp3_out3 = NW.conv(icp3_pool, 1, 1, 64, 1, 1, name='icp3_out3')
            # concat
            icp3_out = NW.concat([icp3_out0,
                                 icp3_out1,
                                 icp3_out2,
                                 icp3_out3], 3, 'icp3_out')
        
        """ 1st classify branch """
        with tf.name_scope('cls1'):
            cls1_pool = NW.avg_pool(icp3_out, 5, 5, 3, 3, padding='VALID', name='cls1_pool')
            cls1_reduction_pose = NW.conv(cls1_pool, 1, 1, 128, 1, 1, name='cls1_reduction_pose')
            cls1_fc1_pose = NW.fc(cls1_reduction_pose, 1024, name='cls1_fc1_pose')
            cls1_fc_pose_xy = NW.fc(cls1_fc1_pose, 2, relu=False, name='cls1_fc_pose_xy')
            cls1_fc_pose_ab = NW.fc(cls1_fc1_pose, 2, relu=False, name='cls1_fc_pose_ab')
            self.layers["cls1_fc_pose_xy"] = cls1_fc_pose_xy
            self.layers["cls1_fc_pose_ab"] = cls1_fc_pose_ab
        
        """ 4st inception layer group """
        print("icp4")
        with tf.name_scope('icp4'):
            # branch 0
            icp4_out0 = NW.conv(icp3_out, 1, 1, 160, 1, 1, name='icp4_out0')
            # branch 1
            icp4_reduction1 = NW.conv(icp3_out, 1, 1, 112, 1, 1, name='icp4_reduction1')
            icp4_out1 = NW.conv(icp4_reduction1, 3, 3, 224, 1, 1, name='icp4_out1')
            # branch 2
            icp4_reduction2 = NW.conv(icp3_out, 1, 1, 24, 1, 1, name='icp4_reduction2')
            icp4_out2 = NW.conv(icp4_reduction2, 5, 5, 64, 1, 1, name='icp4_out2')
            # branch 3
            icp4_pool = NW.max_pool(icp3_out, 3, 3, 1, 1, name='icp4_pool')
            icp4_out3 = NW.conv(icp4_pool, 1, 1, 64, 1, 1, name='icp4_out3')
            # concat
            icp4_out = NW.concat([icp4_out0,
                                  icp4_out1,
                                  icp4_out2,
                                  icp4_out3],3, name='icp4_out')

        """ 5st inception layer group """
        print("icp5")
        with tf.name_scope('icp5'):
            # branch 0
            icp5_out0 = NW.conv(icp4_out, 1, 1, 128, 1, 1, name='icp5_out0')
            # branch 1
            icp5_reduction1 = NW.conv(icp4_out, 1, 1, 128, 1, 1, name='icp5_reduction1')
            icp5_out1 = NW.conv(icp5_reduction1, 3, 3, 256, 1, 1, name='icp5_out1')
            # branch 2
            icp5_reduction2 = NW.conv(icp4_out,1, 1, 24, 1, 1, name='icp5_reduction2')
            icp5_out2 = NW.conv(icp5_reduction2, 5, 5, 64, 1, 1, name='icp5_out2')
            # branch 3
            icp5_pool = NW.max_pool(icp4_out,3, 3, 1, 1, name='icp5_pool')
            icp5_out3 = NW.conv(icp5_pool, 1, 1, 64, 1, 1, name='icp5_out3')
            # concat
            icp5_out = NW.concat([icp5_out0, 
                                  icp5_out1, 
                                  icp5_out2, 
                                  icp5_out3], 3, name='icp5_out')
        
        """ 6st inception layer group """
        print("icp6")
        with tf.name_scope('icp6'):
            # branch 0
            icp6_out0 = NW.conv(icp5_out, 1, 1, 112, 1, 1, name='icp6_out0')
            # branch 1
            icp6_reduction1 = NW.conv(icp5_out, 1, 1, 144, 1, 1, name='icp6_reduction1')
            icp6_out1 = NW.conv(icp6_reduction1, 3, 3, 288, 1, 1, name='icp6_out1')
            # branch 2
            icp6_reduction2 = NW.conv(icp5_out, 1, 1, 32, 1, 1, name='icp6_reduction2')
            icp6_out2 = NW.conv(icp6_reduction2, 5, 5, 64, 1, 1, name='icp6_out2')
            # branch 3
            icp6_pool = NW.max_pool(icp5_out,3, 3, 1, 1, name='icp6_pool')
            icp6_out3 = NW.conv(icp6_pool, 1, 1, 64, 1, 1, name='icp6_out3')
            # concat
            icp6_out = NW.concat([icp6_out0,
                                  icp6_out1,
                                  icp6_out2,
                                  icp6_out3], 3, name='icp6_out')

        """ 2nd classify branch """
        with tf.name_scope('cls2'):
            cls2_pool = NW.avg_pool(icp6_out, 5, 5, 3, 3, padding='VALID', name='cls2_pool')
            cls2_reduction_pose = NW.conv(cls2_pool, 1, 1, 128, 1, 1, name='cls2_reduction_pose')
            cls2_fc1 = NW.fc(cls2_reduction_pose, 1024, name='cls2_fc1')
            cls2_fc_pose_xy = NW.fc(cls2_fc1, 2, relu=False, name='cls2_fc_pose_xy')
            cls2_fc_pose_ab = NW.fc(cls2_fc1, 2, relu=False, name='cls2_fc_pose_ab')
            self.layers["cls2_fc_pose_xy"] = cls2_fc_pose_xy
            self.layers["cls2_fc_pose_ab"] = cls2_fc_pose_ab

        """ 7st inception layer group """
        print("icp7")
        with tf.name_scope('icp7'):
            # branch 0
            icp7_out0 = NW.conv(icp6_out, 1, 1, 256, 1, 1, name='icp7_out0')
            # branch 1
            icp7_reduction1 = NW.conv(icp6_out, 1, 1, 160, 1, 1, name='icp7_reduction1')
            icp7_out1 = NW.conv(icp7_reduction1, 3, 3, 320, 1, 1, name='icp7_out1')
            # branch 2
            icp7_reduction2 = NW.conv(icp6_out, 1, 1, 32, 1, 1, name='icp7_reduction2')
            icp7_out2 = NW.conv(icp7_reduction2, 5, 5, 128, 1, 1, name='icp7_out2')
            # branch 3
            icp7_pool = NW.max_pool(icp6_out, 3, 3, 1, 1, name='icp7_pool')
            icp7_out3 = NW.conv(icp7_pool, 1, 1, 128, 1, 1, name='icp7_out3')
            # concat
            icp7_out = NW.concat([icp7_out0,
                                  icp7_out1,
                                  icp7_out2,
                                  icp7_out3], 3, name='icp7_out')

        """ 8st inception layer group """
        print("icp8")
        with tf.name_scope('icp8'):
            icp8_in = NW.max_pool(icp7_out, 3, 3, 2, 2, name='icp8_in')
            # branch 0
            icp8_out0 = NW.conv(icp8_in, 1, 1, 256, 1, 1, name='icp8_out0')
            # branch 1
            icp8_reduction1 = NW.conv(icp8_in, 1, 1, 160, 1, 1, name='icp8_reduction1')
            icp8_out1 = NW.conv(icp8_reduction1, 3, 3, 320, 1, 1, name='icp8_out1')
            # branch 2
            icp8_reduction2 = NW.conv(icp8_in, 1, 1, 32, 1, 1, name='icp8_reduction2')
            icp8_out2 = NW.conv(icp8_reduction2, 5, 5, 128, 1, 1, name='icp8_out2')
            # branch 3
            icp8_pool = NW.max_pool(icp8_in, 3, 3, 1, 1, name='icp8_pool')
            icp8_out3 = NW.conv(icp8_pool, 1, 1, 128, 1, 1, name='icp8_out3')
            # concat
            icp8_out = NW.concat([icp8_out0,
                                  icp8_out1,
                                  icp8_out2,
                                  icp8_out3], 3, name='icp8_out')
        
        """ 9st inception layer group """
        print("icp9")
        with tf.name_scope('icp9'):
            # branch 0
            icp9_out0 = NW.conv(icp8_out, 1, 1, 384, 1, 1, name='icp9_out0')
            # branch 1
            icp9_reduction1 = NW.conv(icp8_out, 1, 1, 192, 1, 1, name='icp9_reduction1')
            icp9_out1 = NW.conv(icp9_reduction1, 3, 3, 384, 1, 1, name='icp9_out1')
            # branch 2
            icp9_reduction2 = NW.conv(icp8_out, 1, 1, 48, 1, 1, name='icp9_reduction2')
            icp9_out2 = NW.conv(icp9_reduction2, 5, 5, 128, 1, 1, name='icp9_out2')
            # branch 3
            icp9_pool = NW.max_pool(icp8_out, 3, 3, 1, 1, name='icp9_pool')
            icp9_out3 = NW.conv(icp9_pool, 1, 1, 128, 1, 1, name='icp9_out3')
            # concat
            icp9_out = NW.concat([icp9_out0,
                                  icp9_out1,
                                  icp9_out2,
                                  icp9_out3], 3, name='icp9_out')

        """ 3rd classify branch """
        with tf.name_scope('cls3'):
            cls3_pool = NW.avg_pool(icp9_out, 7, 7, 1, 1, padding='VALID', name='cls3_pool')
            cls3_fc1_pose = NW.fc(cls3_pool, 2048, name='cls3_fc1_pose')
            cls3_fc_pose_xy = NW.fc(cls3_fc1_pose, 2, relu=False, name='cls3_fc_pose_xy')
            cls3_fc_pose_ab = NW.fc(cls3_fc1_pose, 2, relu=False, name='cls3_fc_pose_ab')
            self.layers["cls3_fc_pose_xy"] = cls3_fc_pose_xy
            self.layers["cls3_fc_pose_ab"] = cls3_fc_pose_ab
            
            
    def load_initial_weights(self, session, SKIP_LAYER=[]):
        """
        load pretraining weights except the layers in self.skip_layers into memary
        the input weights is a dict-type data, 
        each key is the name of a layer, such as "conv1", "fc1"m etc
        and their value is also a dict has key: "weight" and "biases", eg.
        w = layer_params["conv1"]["weights"]
        b = layer_params["conv1"]["biases"]
        """
        self.SKIP_LAYER = SKIP_LAYER
        layer_params = np.load(self.WEIGHTS_PATH, encoding = "latin1").item()
        
        # Loop over all layer names stored in the weights dict
        for op_name in layer_params:
            # Check if the layer is one of the layers that should be reinitialized
            if op_name not in self.SKIP_LAYER:
                with tf.variable_scope(op_name, reuse = True):
                    # Loop over list of weights/biases and assign them to their corresponding tf variable
                    for key in layer_params[op_name]:
#                         print("load layer params:%s" % op_name)
                        data = layer_params[op_name][key]
                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable = False)
                            session.run(var.assign(data))
                        # Weights
                        else:
                            var = tf.get_variable('weights', trainable = False)
                            session.run(var.assign(data))

if __name__=="__main__":
    pass

