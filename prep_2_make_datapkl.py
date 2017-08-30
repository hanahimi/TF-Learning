#-*-coding:UTF-8-*-
'''
Created on 2016-8-17

@author: hanahimi
'''
import os
import dataio
import cv2
import numpy as np
from datatype import DataPackage

class ImgDataset:
    def __init__(self,index_dir, IMAGE_SIZE, CLASS_NUM):
        pass
        self.index_dir = index_dir
        self.img_size = IMAGE_SIZE
        self.class_num = CLASS_NUM
        self.makedataset()
        
    def _getfilelist(self, name="train.txt"):
        img_list = []
        with open(os.path.join(self.index_dir,name)) as f:
            for line in f.readlines():
                items = line.strip().split("\t")
                img_list.append( (items[0],int(items[1])) )
        return img_list
        
    def makedataset(self):
        self.height = self.img_size
        self.width = self.img_size
        print("make training set")
        img_list = self._getfilelist("train.txt")
        self.train = DataPackage(img_list,self.img_size,self.class_num)
        for path,_ in self.train.img_list:
            im = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2GRAY)
            im = cv2.resize(im,(self.height, self.width))
            im = np.float32(im)
            im = np.reshape(im, (self.height*self.width))
            self.train.mean_image[:] += im[:]
        self.train.mean_image[:] /= self.train.num
        
        print("make val set")
        img_list = self._getfilelist("val.txt")
        self.val  = DataPackage(img_list,self.img_size,self.class_num)
        self.val.mean_image[:] = self.train.mean_image[:]
        
        print("make test set")
        img_list = self._getfilelist("test.txt")
        self.test  = DataPackage(img_list,self.img_size,self.class_num)
        self.test.mean_image[:] = self.train.mean_image[:]
        
    
if __name__=="__main__":
    pass
    # usr input
    root_dir = r"D:\vision studio\workspace\dataset\BSD\BSD_train(3)"
    IMSIZE = 32
    pkl_name = "BSD_C3"
    # make pkl
    image_dir = os.path.join(root_dir,"ImageFiles")
    index_dir = os.path.join(root_dir,"IndexFiles")
    CLASS_NUM = len(os.listdir(image_dir))
    mydataset = ImgDataset(index_dir, IMSIZE, CLASS_NUM)
    dataio.store_pickle(r'dataset\%s_%d.pkl'%(pkl_name, IMSIZE), mydataset)
    print("dataset_index and mean_img are pickled")








