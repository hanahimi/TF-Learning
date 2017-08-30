#-*-coding:UTF-8-*-
'''
Created on 2016-7-20

@author: hanahimi
'''
from dataio import get_filelist

import random
import os

def write_list_index(src_pathlist, dst_txt_path):
    with open(dst_txt_path, "w+") as f:
        for item in src_pathlist:
            if "/" in item:
                label = item.split("/")[-2]
            else:
                label = item.split("\\")[-2]
            label = label.split("_")[0]
            line = item+"\t"+label+"\n"
            f.write(line)


class DataIndex:
    def __init__(self,train_rate=0.5, val_rate=0.2, test_rate=0.3):
        self.path_list = []
        self.total_num = 0
        
        denom = train_rate + val_rate + test_rate
        self.train_rate = 1.0 * abs(train_rate) / denom
        self.val_rate = 1.0 * abs(val_rate) / denom
        self.test_rate = 1.0 * abs(test_rate) / denom
        
        self.train_set = []
        self.val_set = []
        self.test_set = []
        
    def setdata(self, rootdir):
        self.path_list = get_filelist(rootdir,'.jpg','.png','.bmp')
        self.total_num = len(self.path_list)
        random.shuffle(self.path_list)
        train_offset = int(self.total_num*self.train_rate)
        val_offset = int(self.total_num*(self.train_rate+self.val_rate))
        test_offset = int(self.total_num*self.test_rate)
        
        self.train_set = self.path_list[:train_offset]
        self.val_set = self.path_list[train_offset:val_offset]
        self.test_set = self.path_list[-test_offset:]

    def exntdata(self, dataindex):
        self.train_set.extend(dataindex.train_set)
        self.val_set.extend(dataindex.val_set)
        self.test_set.extend(dataindex.test_set)
        random.shuffle(self.train_set)
        random.shuffle(self.val_set)
        random.shuffle(self.test_set)
        self.total_num = len(self.train_set) + len(self.val_set) + len(self.test_set)

    
    def outputIndex(self, txtdir):
        write_list_index(self.train_set, os.path.join(txtdir,"train.txt"))
        write_list_index(self.val_set, os.path.join(txtdir,"val.txt"))
        write_list_index(self.test_set, os.path.join(txtdir,"test.txt"))
        write_list_index(self.train_set+self.val_set, os.path.join(txtdir,"trainval.txt"))
        write_list_index(self.val_set+self.test_set, os.path.join(txtdir,"testval.txt"))


    
def make_index(imgdir, txtdir):
    labels = os.listdir(imgdir)
    imgpath_set = {}
    for label in labels:
        tmp_dir = os.path.join(imgdir, label)
        data_in_label = DataIndex()
        data_in_label.setdata(tmp_dir)
        imgpath_set[label] = data_in_label
        
    total_data = DataIndex()
    for label in labels:
        total_data.exntdata(imgpath_set[label])
    total_data.outputIndex(txtdir)
    
if __name__=="__main__":
    pass
    imgdir = r"D:\vision studio\workspace\dataset\BSD\BSD_train(3)\ImageFiles"
    txtdir = r"D:\vision studio\workspace\dataset\BSD\BSD_train(3)\IndexFiles"
    if not os.path.exists(txtdir):
        os.makedirs(txtdir)
    make_index(imgdir, txtdir)
    
    
    
    