#-*-coding:UTF-8-*-
'''
Created on 2016-8-9

@author: hanahimi
'''
import numpy as np
from dataio import load_pickle
from prep_2_make_datapkl import DataPackage, ImgDataset



def convert_header(fpath, nets):
    with open(fpath,'w+') as f:
        f.write("#ifndef __CNN_MODEL_PARAMS_H\n")
        f.write("#define __CNN_MODEL_PARAMS_H\n\n\n")
        f.write('#define CNN_PARAMS_VER_STR\n')
        
        for key in nets.keys():
            if len(nets[key].shape)==1:
                f.write("extern int l32%sNum;\n" % key)
            elif len(nets[key].shape)==2:
                f.write("extern int l32%sInNum;\n" % key)
                f.write("extern int l32%sOutNum;\n" % key)
            elif len(nets[key].shape)==4:
                f.write("extern int l32%sHeight;\n" % key)
                f.write("extern int l32%sWidth;\n" % key)
                f.write("extern int l32%sChannel;\n" % key)
                f.write("extern int l32%sNum;\n" % key)
            f.write("extern float af32%s[];\n" % key)
            f.write("\n\n")
        
        f.write("extern float af32MeanImage[];\n\n")
        f.write("#endif\n\n")
        

def convert_cpp(fpath, array, pre_fix):
    arr_shape = array.shape
    if not (1<=len(arr_shape)<=4):
        return
    else:
        if len(arr_shape)==1:
            with open(fpath,'w+') as f:
                f.write("int l32%sNum=%d;\n" % (pre_fix,arr_shape[0]))
                f.write("float af32%s[%d] = {" % (pre_fix,arr_shape[0]))
                items = []
                for i in range(arr_shape[0]):
                    if i%10==0:
                        items.append("\n% 9.6ff" % array[i])
                    else:
                        items.append("% 9.6ff" % array[i])
                items = ",".join(items)
                f.write(items)
                f.write("};\n")
        elif len(arr_shape)==2:
            with open(fpath,'w+') as f:
                f.write("int l32%sInNum=%d;\n" % (pre_fix,arr_shape[0]))
                f.write("int l32%sOutNum=%d;\n" % (pre_fix,arr_shape[1]))
                f.write("float af32%s[%d] = {\n" % (pre_fix,arr_shape[0]*arr_shape[1]))
                for row in range(arr_shape[0]):
                    items = []
                    for col in range(arr_shape[1]):
                        items.append("% 9.6ff" % array[row][col])
                    items = ",".join(items)
                    if row%10==0:
                        f.write("\n//"+"-"*int(col/2)+(" line: % 4d " % row) +"-"*int((col/2))+"\n")
                    if row != arr_shape[0]-1:
                        f.write(items+",\n")
                    else:
                        f.write(items+"\n")
                f.write("};\n")
        elif len(arr_shape)==4:
            with open(fpath,'w+') as f:
                row,col,chn,num = arr_shape
                f.write("int l32%sHeight=%d;\n" % (pre_fix,row))
                f.write("int l32%sWidth=%d;\n" % (pre_fix,col))
                f.write("int l32%sChannel=%d;\n" % (pre_fix,chn))
                f.write("int l32%sNum=%d;\n" % (pre_fix,num))
                f.write("float af32%s[%d] = {\n" % (pre_fix,num*chn*row*col))
                
                for n in range(num):
                    f.write("\n// output-%d\t%dx[%d*%d]\n" % (n,chn,row,col))
                    for c in range(chn):
                        f.write("//=============kernel-%d, channel-%d==============" % (n,c))
                        one_filter = []
                        for i in range(row):
                            for j in range(col):
                                if j == 0:
                                    one_filter.append("\n% 9.6ff" % array[i][j][c][n])
                                else:
                                    one_filter.append("% 9.6ff" % array[i][j][c][n])
                        if c != chn-1:
                            one_filter = ",".join(one_filter)+",\n"
                        else:
                            if n != num-1:
                                one_filter = ",".join(one_filter)+",\n"
                            else:
                                one_filter = ",".join(one_filter)+"\n"
                        f.write(one_filter)
                        
                f.write("};\n")

def write_mean_image(im):
    with open("convert_cpp/cnn_mean_image.cpp","w+") as f:
        n = im.shape[0]
        f.write('#include "cnn_params.h"\n\n')
        f.write("float af32MeanImage[%d]={\n" % n)
        for i in range(n):
            if i % 32 == 0:
                f.write("\n")
            f.write("% 9.6ff," % im[i])
        f.write("};\n")
            
def convert_cpp_models(model_name,input_size,data_name):
    weights_path = "./save_weights/%s_%d.pkl" % (model_name,input_size)
    nets = load_pickle(weights_path)
    # write cpp
    for key in nets.keys():
        weight = nets[key]
        print(key,weight.shape)
        fpath = "convert_cpp/cnn_"+key+".cpp"
        convert_cpp(fpath, weight, key)
    
    # write_header
    fpath = "convert_cpp/cnn_params.h"
    convert_header(fpath, nets)
    
    # write mean image
    dataset_path = "./dataset/%s_%d.pkl" % (data_name,input_size)
    print(dataset_path)
    dataset = load_pickle(dataset_path)
    mean_img = dataset.train.mean_image
    print("mean image",mean_img.shape)
    write_mean_image(mean_img)
    
    

def main():
    model_name = "leNet5-sp3"
    input_size = 32
    convert_cpp_models(model_name, input_size)
    
    
    
if __name__=="__main__":
    pass
    main()


    
    