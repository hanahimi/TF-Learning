#-*-coding:UTF-8-*-
'''
Created on 2017-10-15--9:56:58 a.m
author: Gary-W
ref
http://blog.csdn.net/Run_it_faraway/article/details/76100748

解析caffe网络结构并将npy文件中的参数(W,b)保存到caffemodel中
该脚本将解析prototxt中的网络结构，从npy中查找名称对应的参数进行写入

caffe的网络结构文件需要包含npy中的自动

caffe 网络权重 与 tf 相互转换,使用layer的名字进行对应
注意到：

tensorflow 权重参数的维度对应为 [filter_height, filter_width, in_channels, out_channels]
                                     0               1            2            3

caffe 权重参数的维度对应为     [out_channels, in_channels, filter_height, filter_width] 
                                 3             2            0              1
eg. caffe:  a convolution layer with 96 filters of 11 x 11 spatial dimension and 3 inputs the blob is:
                            96 x 3 x 11 x 11
因此需要将tf-npy的权重的通道维度进行转置, fc 层同理

'''
import os
import caffe
import numpy as np

caffe_net = 'data/src.prototxt'
dst_caffe_weights = 'data/src_npy2caffe.caffemodel'
npy_path = 'data/src.npy'

# 载入网络，列出各个层的名字
caffe_model = caffe.Net(caffe_net, caffe.TEST)

# 查看所有的 blobs
# for blobs in caffe_model.blobs.keys():
#     print(blobs)
# 查看所有的 layer 类型
# for i, layer in enumerate(caffe_model.layers):
#     layer_name = caffe_model._layer_names[i]
#     print(layer + " : " + layer.type)
# 查看所有带权重参数的层
# for params in caffe_model.params.keys():
#     print(params)
caffe_params = [param for param in caffe_model.params.keys()]
caffe_dict = {caffe_model._layer_names[i]: layer.type \
              for i, layer in enumerate(caffe_model.layers)\
              if caffe_model._layer_names[i] in caffe_params}


# 载入npy文件并选择保存的参数
npy_params = np.load(npy_path, encoding = "latin1").item()
SKIP_LAYER = []
for i, layer in enumerate(caffe_model.layers):
    layer_name = caffe_model._layer_names[i]
    if layer_name in caffe_params and layer_name not in SKIP_LAYER:
#         if layer_name in npy_params:
            print("processing " + layer_name + " : " + layer.type)

            if layer.type == "Convolution":
                weights = npy_params[layer_name]["weights"]
                weights = weights.transpose((3,2,0,1))
                layer.blobs[0].data[:] = weights[:]
                print("\t weights:",weights.shape," ==> ",layer.blobs[0].data.shape)

                biases = npy_params[layer_name]["biases"]
                layer.blobs[1].data[:] = biases[:]
                print("\t biases:",biases.shape," ==> ",layer.blobs[1].data.shape)

            elif layer.type == "InnerProduct":
                weights = npy_params[layer_name]["weights"]
                weights = weights.transpose((1, 0))
                layer.blobs[0].data[:] = weights[:]
                print("\t weights:",weights.shape," ==> ",layer.blobs[0].data.shape)

                biases = npy_params[layer_name]["biases"]
                layer.blobs[1].data[:] = biases[:]
                print("\t biases:",biases.shape," ==> ",layer.blobs[1].data.shape)

caffe_model.save(dst_caffe_weights)


def formatSize(bytes):
    # 字节bytes转化kb\m\g
    try:
        bytes = float(bytes)
        kb = bytes / 1024
    except:
        print("传入的字节格式不对")
        return "Error"
    if kb >= 1024:
        M = kb / 1024
        if M >= 1024:
            G = M / 1024
            return "%fG" % (G)
        else:
            return "%fM" % (M)
    else:
        return "%fkb" % (kb)

print("save caffe model, size = ", formatSize(os.path.getsize(dst_caffe_weights)))
 
if __name__=="__main__":
    pass

