# encoding:utf-8
'''
Created on 2017年1月28日
@author: Ayumi Phoenix
'''
import pickle

def load(filename):
    f = open(filename, "rb")
    d = pickle.load(f)
    f.close()
    return d

def get_chunk(samples, labels, chunkSize):
    '''
    这个函数是一个迭代器/生成器，用于每一次只得到 chunkSize 这么多的数据
    用于 for loop， just like range() function
    Args:
        samples: 样本数据集 array[in]
        labels:    样本标注 array[in]
    '''
    if len(samples) != len(labels):
        raise Exception('Length of samples and labels must equal')
    stepStart = 0    # initial step
    i = 0
    while stepStart < len(samples):
        stepEnd = stepStart + chunkSize
        if stepEnd < len(samples):
            yield i, samples[stepStart:stepEnd], labels[stepStart:stepEnd]
            i += 1
        stepStart = stepEnd

if __name__=="__main__":
    pass
    
    