#-*-coding:UTF-8-*-
'''
Created on 2017��8��31��-����1:20:16
author: Gary-W

����caffe����ṹ��������� W, b��pickle��
���л���Ϊ py2.7
ref:
http://blog.csdn.net/u011933123/article/details/53589354

caffe ����Ȩ�� �� tf �໥ת��,ʹ��layer�����ֽ��ж�Ӧ
ע�⵽��
tensorflow Ȩ�ز�����ά�ȶ�ӦΪ [filter_height, filter_width, in_channels, out_channels]
                                     2               3            1            0

caffe Ȩ�ز�����ά�ȶ�ӦΪ     [out_channels, in_channels, filter_height, filter_width] 
                                 0             1            2              3
eg. caffe:  a convolution layer with 96 filters of 11 x 11 spatial dimension and 3 inputs the blob is:
                            96 x 3 x 11 x 11


�����Ҫ��ĳһ����Ȩ�ص�ͨ��ά�Ƚ���ת��
fc ��ͬ��

��ˣ��ù��߼��й涨���ս����õ���Ȩ�أ������� [filter_height, filter_width, in_channels, out_channels]�ķ�ʽ���д洢
�ж�Ӧ�����绷������ת��

'''
import caffe
from util.dataio import store_pickle

DEBUG = True

caffe_net = 'data/src.prototxt'
caffe_weights = 'data/src.caffemodel'
save_path = 'data/src.pkl'

caffe_model = caffe.Net(caffe_net, caffe_weights, caffe.TEST)

# ��¼��Ҫ���б���������ؼ���
name_keyword = ["Convolution", "InnerProduct"]

layer_params = {}
for i, layer in enumerate(caffe_model.layers):

    layer_name = caffe_model._layer_names[i]
    print(layer_name+ "is"+ layer.type)

    if layer.type == "Convolution":
        layer_params[layer_name] = {}
        # ע����bolbs.data��һ��ָ�룬ֻ���Է���������
        # ���������copy�����ſɷ������ǵ�ֵ
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
store_pickle(save_path, layer_params)

if __name__=="__main__":
    pass

