#-*-coding:UTF-8-*-
'''
Created on 2016-7-14

@author: hanahimi
'''
import dataio
import cv2
import os
import random
import numpy as np

def is_flip():
    if random.randint(1,10000)<2000:
        return True
    else:
        return False
    
class DataPackage:
    def __init__(self, img_list, img_size, class_num):
        
        self.img_list = img_list
        self.labels = [items[1] for items in self.img_list]
        self.num = len(self.img_list)
        self.class_num = class_num

        self.height = img_size
        self.width = img_size
        # 1d-array
        self.mean_image = np.zeros((self.height*self.width), np.float64)

        self.images = []
        self.onehotlabels = []
        self.fixlabels = [] # in case labels has been shuffled
        self.set_all_batch = False
    
        self.current_batch_idx = 0
         
    def all_batches(self,onehot=True):
        if not self.set_all_batch:
            self.set_all_batch = True
            for i in range(self.num):
                path, label = self.img_list[i]
                im = cv2.cvtColor(cv2.imread(path),cv2.COLOR_RGB2GRAY)
                im = cv2.resize(im,(self.height, self.width))
                im = np.float32(im)
                im = np.reshape(im, (self.height*self.width))
                im -= self.mean_image
                self.images.append(im)
                
                label_oh = np.zeros(self.class_num)
                label_oh[label] = 1
                self.fixlabels.append(label)
                self.onehotlabels.append(label_oh)
                
        if onehot:
            return [self.images, self.onehotlabels]
        else:
            return [self.images, self.fixlabels]
    
    
    def next_batch(self, batch_size=50, onehot=True):
        if self.current_batch_idx + batch_size < self.num:
            self.current_batch_idx += batch_size
        else:
            # re-shuffle data, sample again...
            random.shuffle(self.img_list)
            self.current_batch_idx = batch_size
        
        op = self.current_batch_idx - batch_size
        ed = self.current_batch_idx
        
        data_list = []
        label_list = []
        for i in range(op,ed):
            path, label = self.img_list[i]
            im = cv2.cvtColor(cv2.imread(path),cv2.COLOR_RGB2GRAY)
            im = cv2.resize(im,(self.height, self.width))
            im = np.float32(im)
            im = np.reshape(im, (self.height*self.width))
            im -= self.mean_image
            data_list.append(im)
            if onehot:
                label_oh = np.zeros(self.class_num)
                label_oh[label] = 1
                label_list.append(label_oh)
            else:
                label_list.append(label)
            
        return [data_list, label_list]
        
        
    
if __name__=="__main__":
    pass
    
    
    
    