#-*-coding:UTF-8-*-
'''
Created on 2016-6-17
@author: hanahimi
'''
import os
import pickle

def get_filelist(root_dir, *subfixs):
    """
    sample:
    get_filelist(r'../daytime_adaboost/pos',".png",".jpg")
    """
    p = []
    for subfix in subfixs:
        p.extend([os.path.join(root_dir,f) for f in os.listdir(root_dir) if f.endswith(subfix)])
    return p
    
def store_pickle(path, obj):
    try:
        with open(path, 'wb') as fw:
            pickle.dump(obj, fw)
    except IOError as ioerr:    
        print( "IO Error:"+str(ioerr)+"in:\n"+path)

def load_pickle(path):
    try:
        with open(path, 'rb') as fr:
            obj = pickle.load(fr)
            return obj
    except IOError as ioerr:    
        print("IO Error:"+str(ioerr)+"in:\n"+path)


if __name__=="__main__":
    pass
    
    