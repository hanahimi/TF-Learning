# encoding:utf-8
'''
Created on 2017年1月28日
@author: Ayumi Phoenix
认识数据集 http://ufldl.stanford.edu/housenumbers/
数据处理函数会作为该实验中的工具使用

'''
from scipy.io import loadmat as load
import numpy as np
import pickle

def reformat(samples, labels):
    # 改变原始数据的形状
    #  0       1       2      3          3       0       1      2
    # (图片高，图片宽，通道数，图片数) -> (图片数，图片高，图片宽，通道数)
    new = np.transpose(samples, (3, 0, 1, 2)).astype(np.float32)

    # labels 变成 one-hot encoding, [2] -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    # digit 0 , represented as 10
    # labels 变成 one-hot encoding, [10] -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    labels = np.array([x[0] for x in labels])    # slow code, whatever
    one_hot_labels = []
    for num in labels:
        one_hot = [0.0] * 10
        if num == 10:
            one_hot[0] = 1.0
        else:
            one_hot[num] = 1.0
        one_hot_labels.append(one_hot)
    labels = np.array(one_hot_labels).astype(np.float32)
    return new, labels

def normalize(samples):
    '''
    灰度化: 从三色通道 -> 单色通道     省内存 + 加快训练速度
    (R + G + B) / 3
    将图片从 0 ~ 255 线性映射到 -1.0 ~ +1.0
    @samples: numpy array
    '''
    a = np.add.reduce(samples, keepdims=True, axis=3)  # shape (图片数，图片高，图片宽，通道数)
    a = a/3.0
    return a/128.0 - 1.0


def save_new_data(filename, samples,labels):
    f = open(filename,'wb')
    d = {"X":samples, "y":labels}
    pickle.dump(d,f)
    f.close()

def load_pickle(filename):
    f = open(filename, "rb")
    d = pickle.load(f)
    f.close()
    return d


if __name__=="__main__":
    pass
    train = load('../../data/train_32x32.mat')
    test = load('../../data/test_32x32.mat')
    print('Train Samples Shape:', train['X'].shape)
    print('Train  Labels Shape:', train['y'].shape)

    n_train_samples, _train_labels = reformat(train['X'], train['y'])
    n_test_samples, _test_labels = reformat(test['X'], test['y'])
    print('Train Samples Shape:', n_train_samples.shape)
    print('Train  Labels Shape:', _train_labels.shape)
    n_train_samples = normalize(n_train_samples)
    n_test_samples = normalize(n_test_samples)
    save_new_data("../../data/ntrain.pickle", n_train_samples,_train_labels)
    save_new_data("../../data/ntest.pickle", n_test_samples,_test_labels)

    

    
    