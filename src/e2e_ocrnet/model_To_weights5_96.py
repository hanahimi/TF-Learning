import tensorflow as tf
from e2e_ocrnet.model5_96 import Model
import cv2
import io
import numpy 

tf.app.flags.DEFINE_string('images', r'C:\WTF\SVHNClassifier\images\train1.png', 'Path to image file')
tf.app.flags.DEFINE_string('restore_checkpoint', r"D:\conda_env\workspace\py\TF-Learning\src\e2e_ocrnet\logs_saic_0908\model.ckpt-3100",
                            'Path to restore checkpoint ')
FLAGS = tf.app.flags.FLAGS
save_root = r'D:\conda_env\workspace\py\TF-Learning\src\e2e_ocrnet\convert_weight'
    
def convert_header(fpath, nets):
    with open(fpath,'w+') as f:
        f.write("#ifndef __OCR_CNN_MODEL_PARAMS_H\n")
        f.write("#define __OCR_CNN_MODEL_PARAMS_H\n\n\n")
        f.write('#define OCR_CNN_PARAMS_VER_STR\n')
         
        for key in nets.keys():
            if len(nets[key].shape)==1:
                f.write("extern int Ocr_l32%sNum;\n" % key)
            elif len(nets[key].shape)==2:
                f.write("extern int Ocr_l32%sInNum;\n" % key)
                f.write("extern int Ocr_l32%sOutNum;\n" % key)
            elif len(nets[key].shape)==4:
                f.write("extern int Ocr_l32%sHeight;\n" % key)
                f.write("extern int Ocr_l32%sWidth;\n" % key)
                f.write("extern int Ocr_l32%sChannel;\n" % key)
                f.write("extern int Ocr_l32%sNum;\n" % key)
            f.write("extern float Ocr_af32%s[];\n" % key)
            f.write("\n\n")
         
#         f.write("extern float Ocr_af32MeanImage[];\n\n")
        f.write("#endif\n\n")
         
 
def convert_cpp(fpath, array, pre_fix):
    arr_shape = array.shape
    if not (1<=len(arr_shape)<=4):
        return
    else:
        if len(arr_shape)==1:
            with open(fpath,'w+') as f:
                f.write("int Ocr_l32%sNum=%d;\n" % (pre_fix,arr_shape[0]))
                f.write("float Ocr_af32%s[%d] = {" % (pre_fix,arr_shape[0]))
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
                f.write("int Ocr_l32%sInNum=%d;\n" % (pre_fix,arr_shape[0]))
                f.write("int Ocr_l32%sOutNum=%d;\n" % (pre_fix,arr_shape[1]))
                f.write("float Ocr_af32%s[%d] = {\n" % (pre_fix,arr_shape[0]*arr_shape[1]))
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
                f.write("int Ocr_l32%sHeight=%d;\n" % (pre_fix,row))
                f.write("int Ocr_l32%sWidth=%d;\n" % (pre_fix,col))
                f.write("int Ocr_l32%sChannel=%d;\n" % (pre_fix,chn))
                f.write("int Ocr_l32%sNum=%d;\n" % (pre_fix,num))
                f.write("float Ocr_af32%s[%d] = {\n" % (pre_fix,num*chn*row*col))
                
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
    with open("convert_cpp/corner_cnn_mean_image.cpp","w+") as f:
        n = im.shape[0]
        f.write('#include "corner_cnn_params.h"\n\n')
        f.write("float Corner_af32MeanImage[%d]={\n" % n)
        for i in range(n):
            if i % 32 == 0:
                f.write("\n")
            f.write("% 9.6ff," % im[i])
        f.write("};\n")
             
def convert_cpp_models(nets):
    # write cpp
    for key in nets.keys():
        weight = nets[key]
        print(key,weight.shape)
        fpath = save_root+"\\Ocr_cnn_"+key+".cpp"
        convert_cpp(fpath, weight, key)
    # write_header
    fpath = save_root+"\\Ocr_cnn_params.h"
    convert_header(fpath, nets)


def main(_):
    
    path_to_image_file = FLAGS.images
    path_to_restore_checkpoint_file = FLAGS.restore_checkpoint
    image = tf.image.decode_jpeg(tf.read_file(path_to_image_file), channels=3)
#     image = tf.image.resize_images(image, [64, 64])
#     image = cv2.imread(path_to_image_file)
#     image = tf.reshape(image, [64, 64, 3])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
#     image = tf.multiply(tf.subtract(image, 0.5), 2)
    image = tf.image.resize_images(image, [54, 54])
    images = tf.reshape(image, [1, 54, 54, 3])

    length_logits, digits_logits = Model.inference(images, drop_rate=0.0)
    length_predictions = tf.argmax(length_logits, axis=1)
    digits_predictions = tf.argmax(digits_logits, axis=2)
    digits_predictions_string = tf.reduce_join(tf.as_string(digits_predictions), axis=1)
    with tf.Session() as sess:
        restorer = tf.train.Saver()
        restorer.restore(sess, path_to_restore_checkpoint_file)
        params=tf.trainable_variables()
        print("Trainable variables:------------------------")

        for idx, v in enumerate(params):
            print("  param {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))
#             save_path = r'%s\param%d.cpp'%(save_root,idx)
#             Weight=sess.run(params[idx])
#             numpy.savetxt(save_path,Weight,fmt='%s',delimiter=' ', newline='\n')
        nets = {}
      
        conv_num = 5
        fc_num = 7
        for i in range(conv_num):
            key_name_w = 'C%dw'%i
            key_name_b = 'C%db'%i
            nets[key_name_w] = sess.run(params[2*i])
            nets[key_name_b] = sess.run(params[2*i+1])
        for i in range(fc_num):
            key_name_fw = 'F%dw'%(conv_num+i)
            key_name_fb = 'F%db'%(conv_num+i)
            nets[key_name_fw] = sess.run(params[2*conv_num+2*i])
            nets[key_name_fb] = sess.run(params[2*conv_num+2*i+1])
            
        print(nets.keys())
        convert_cpp_models(nets)
            

if __name__ == '__main__':
    tf.app.run(main=main)
        



