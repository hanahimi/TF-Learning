#-*-coding:UTF-8-*-
'''
Created on 2017年6月28日
@author: Ayumi Phoenix
'''
import tensorflow as tf

m1 = tf.constant([[2,2]])
m2 = tf.constant([[3],[3]])
dot_op = tf.mul(m1, m2)
print(dot_op)

sess = tf.Session()




if __name__=="__main__":
    pass
    
    