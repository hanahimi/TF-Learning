#-*-coding:UTF-8-*-
'''
Created on 2016-8-9

@author: hanahimi
'''
import tensorflow as tf
from dataio import *

def saveNets_pkl(nets, model_name, input_size):
    save_pkl_path = ("./save_weights/%s_%d.pkl" % (model_name,input_size))
    save_weights = {}
    saver = tf.train.Saver()
    with tf.Session()  as sess:
        saver.restore(sess, ("./save_ckpt/%s_%d.ckpt" % (model_name,input_size)))
        print("Model restored")
        for key in nets:
            try:
                print(key,":",nets[key].eval().shape)
                save_weights[key] = nets[key].eval()
            except:
                print(key," have no value")
    for k in save_weights:
        print(k)
    store_pickle(save_pkl_path, save_weights)
    print("\nSaved")
        

if __name__=="__main__":
    pass
    
    
    