#-*-coding:UTF-8-*-
'''
Created on 2017-9-26-6:42:10 pm
author: Gary-W
'''
import tensorflow as tf
from vps.network_tookit import NW


class Posenet(object):
    def __init__(self, x, weights_path):
        """
        Inputs:
        x: placeholder for images
        weights_path: path string, path to pretraining weights, a pkl file
        """
        self.X = x
        self.weights_path = weights_path

        # call the create function to build the computational graph of posenet
        self.create()
    
    def create(self):
        """ create pose network (based on googlenet)"""
        
        """ A solo prepressing reduction network in the head """
        conv1 = NW.conv(self.X, 7, 7, 64, 2, 2, name='conv1')
        pool1 = NW.max_pool(conv1, 3, 3, 2, 2, name='pool1')
        norm1 = NW.lrn(pool1, 2, 2e-05, 0.75, name='norm1')
        reduction2 = NW.conv(norm1, 1, 1, 64, 1, 1, name='reduction2')
        conv2 = NW.conv(reduction2, 3, 3, 192, 1, 1,name='conv2')
        norm2 = NW.lrn(conv2, 2, 2e-05, 0.75, name='norm2')
        pool2 = NW.max_pool(norm2, 3, 3, 2, 2, name='pool2')
        
        """ 1st inception layer group """
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
        

        """ 2nd inception layer group """
        # concat
        icp2_in = NW.concat([icp1_out0,
                             icp1_out1,
                             icp1_out2,
                             icp1_out3], 3, 'icp2_in')
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
        icp3_in = NW.max_pool(icp2_out, 3, 3, 2, 2, name='icp3_in')
        
        """ 3rd inception layer group """
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
        
        
    def load_initial_weights(self):
        pass



if __name__=="__main__":
    pass

