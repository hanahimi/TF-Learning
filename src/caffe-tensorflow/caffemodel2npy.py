#-*-coding:UTF-8-*-
'''
Created on 2017年8月31日-下午1:20:16
author: Gary-W

解析caffe网络结构并保存参数 W, b到pickle中
运行环境为 py2.7
ref:
http://blog.csdn.net/u011933123/article/details/53589354

caffe 网络权重 与 tf 相互转换,使用layer的名字进行对应
注意到：
tensorflow 权重参数的维度对应为 [filter_height, filter_width, in_channels, out_channels]
                                     2               3            1            0

caffe 权重参数的维度对应为     [out_channels, in_channels, filter_height, filter_width] 
                                 0             1            2              3
eg. caffe:  a convolution layer with 96 filters of 11 x 11 spatial dimension and 3 inputs the blob is:
                            96 x 3 x 11 x 11


因此需要将某一方的权重的通道维度进行转置
fc 层同理

因此，该工具集中规定最终解析得到的权重，均采用 [filter_height, filter_width, in_channels, out_channels]的方式进行存储
有对应的网络环境自行转置

'''
import caffe
import numpy as np

caffe_net = 'data/src.prototxt'
caffe_weights = 'data/src.caffemodel'
save_path = 'data/src.npy'

caffe_model = caffe.Net(caffe_net, caffe_weights, caffe.TEST)

# 记录需要进行保存的网络层关键字
name_keyword = ["Convolution", "InnerProduct"]

layer_params = {}
for i, layer in enumerate(caffe_model.layers):

    layer_name = caffe_model._layer_names[i]
    print(layer_name+ "is"+ layer.type)

    if layer.type == "Convolution":
        layer_params[layer_name] = {}
        # 注意这bolbs.data是一个指针，只可以方法其属性
        # 必须进行深copy操作才可访问他们的值
        weights = layer.blobs[0].data
        biases = layer.blobs[1].data
        layer_params[layer_name]['weights'] = weights.transpose((2, 3, 1, 0))
        layer_params[layer_name]['biases'] = biases.copy()
        

    elif layer.type == "InnerProduct":
        layer_params[layer_name] = {}
        weights = layer.blobs[0].data
        biases = layer.blobs[1].data
        layer_params[layer_name]['weights'] = weights.transpose((1, 0))
        layer_params[layer_name]['biases'] = biases.copy()

    try:
        print(layer_name + ':'+str(weights.shape)+'==>'+str(layer_params[layer_name]['weights'].shape))
        print(layer_name + ':'+str(biases.shape)+'==>'+str(layer_params[layer_name]['biases'].shape))
    except:
        print("this layer has no weights")
    print("\n")

print("save params")
np.save(save_path, layer_params)


if __name__=="__main__":
    pass

