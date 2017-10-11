#-*-coding:UTF-8-*-
'''
Created on 2017年9月22日-下午8:45:52
author: Gary-W
常用图像增强操作
'''

import numpy as np
import cv2
import random
from skimage import exposure
from PIL import Image 

class ImageTransformer:
    def __init__(self):
        pass
    
    @staticmethod
    def centered_crop(img, output_H=224, output_W=224):
        """ Crop the center ROI from input image
        Args:
          img: ndarray-like, shape = (c,h,w)
          output_H: output size height default(224)
          output_W: output size width default(224)
        Return:
            cropped_img: center image of given size
        """
        rows, cols, _ = img.shape
        new_H = output_H
        new_W = output_W
        if rows > cols:
            new_H = output_W * rows / cols
        else:
            new_W = output_W * cols / rows
        H_offset = int((new_H - output_H) / 2)
        W_offset = int((new_W - output_W) / 2)
        cropped_img = img[H_offset:H_offset + output_H, W_offset:W_offset + output_W]
        return cropped_img
    
    @staticmethod
    def random_crop(image, bbox_left, bbox_top, bbox_width, bbox_height,scale=1.3):
        # 以bbox为中心随机剪裁正方形区域
        # scale表示最大边对应扩展的比例
        image_height, image_width, _ = image.shape
        croped_width = int(round(bbox_width * scale))
        croped_height = int(round(bbox_height * scale))
        
        width_offset_base = croped_width - bbox_width
        height_offset_base = croped_height - bbox_height
        
        width_offset = np.random.randint(width_offset_base/3, width_offset_base+1)
        height_offset = np.random.randint(height_offset_base/3, height_offset_base+1)
       
        cropped_left = bbox_left - width_offset
        cropped_top = bbox_top - height_offset
        # 防止裁剪框溢出
        if cropped_left < 0: cropped_left = 0
        if cropped_left > image_width - croped_width - 1: cropped_left = image_width - croped_width - 1
        if cropped_top < 0: cropped_top = 0
        if cropped_top > image_height - croped_height - 1: cropped_top = image_height - croped_height - 1
        
        image_cropped = image[cropped_top:cropped_top+croped_height, cropped_left:cropped_left+croped_width]
        return image_cropped
    
    @staticmethod
    def random_rotation(image, scale=10):
        # 对图像在指定角度范围(-scale, +scale) deg内进行随机旋转
        rand_deg = (np.random.rand()-0.5) * 2 * scale
        image_rotated = Image.fromarray(image)
        image_rotated = image_rotated.rotate(rand_deg)
        image_rotated = np.asarray(image_rotated)
#         rows, cols, _ = image.shape
#         M = cv2.getRotationMatrix2D((cols/2,rows/2), - rand_deg, 1)
#         image_rotated = cv2.warpAffine(image,M,(cols,rows))
        return image_rotated

    @staticmethod
    def random_contrast_stretching(image, rate=0.5):
        # 随机对图像进行的对比度拉伸(98.5% ± 1.5%, 1.5% ± 1.5%)
        if np.random.random() < rate:
            i98 = 98.5 + (np.random.rand()-0.5)*3 
            i2 = 1.5 + (np.random.rand()-0.5)*3
            p2, p98 = np.percentile(image, (i2, i98))
            image_stretched = exposure.rescale_intensity(image, in_range=(p2, p98))
        else:
            image_stretched = image
        return image_stretched
    
    @staticmethod
    def random_hist_equalization(image, rate):
        # 随机对图像进直方图均衡化(bug)
        if np.random.random() < rate:
            image_eq_hist = exposure.equalize_hist(image)
        else:
            image_eq_hist = image
        return image_eq_hist

    @staticmethod
    def random_equalize_adapthist(image, rate):
        # 随机对图像进直方图自适应均衡化（按通道）
        image_adapteq = np.float64(np.copy(image))
        if np.random.random() < rate:
            if len(image.shape) == 3:
                for i in range(image.shape[2]):
                    image_adapteq[:,:,i] = exposure.equalize_adapthist(image[:,:,i], clip_limit=0.001)
            elif len(image.shape) == 2:
                image_adapteq = exposure.equalize_adapthist(image, clip_limit=0.001)
        else:
            image_adapteq = image
        return image_adapteq
    
    @staticmethod
    def random_exchange_channel(image, rate):
        # 随机对图像进行通道交互
        if np.random.random() < rate:
            image_exchannel = np.copy(image)
            if len(image.shape) == 3:
                ch = [0,1,2]
                random.shuffle(ch)
                for i in range(len(ch)):
                    image_exchannel[:,:,i] = image[:,:,ch[i]]
        else:
            image_exchannel = image
        return image_exchannel

    @staticmethod
    def random_flipLR(image, rate):
        # 随机对图像进行左右翻转
        if np.random.random() < rate:
            image_fliplr = np.copy(image)
            cols = image.shape[1]
            for i in range(cols):
                image_fliplr[:,i] = image[:,cols - i - 1]
        else:
            image_fliplr = image
        return image_fliplr

    @staticmethod
    def random_flipUD(image, rate):
        # 随机对图像进行上下翻转
        if np.random.random() < rate:
            image_flipud = np.copy(image)
            rows = image.shape[1]
            for i in range(rows):
                image_flipud[i,:] = image[rows - i - 1,:]
        else:
            image_flipud = image
        return image_flipud
    
    @staticmethod
    def random_zoom(image, rate):
        # 随机对图像缩放
        return image
    
        
if __name__=="__main__":
    pass
    p = r"E:\OCR-PLD_Dataset\saic_dataset\raw_data_100x100\091701\crop\00134L_264_22_364_122.png"
    image = cv2.imread(p)
    image_adapteq = ImageTransformer.random_rotation(image, 10.)
    cv2.imshow("ASdas", image_adapteq)
    cv2.waitKey(0)




    