#-*-coding:UTF-8-*-
'''
Created on 2017年6月28日
@author: Ayumi Phoenix
'''
import tensorflow as tf

m1 = tf.constant([[2,2]])
m2 = tf.constant([[3],[3]])
dot_op = tf.matmul(m1, m2)

print(dot_op)   # just print op's shape not result

# method 1 use version
sess = tf.Session()
result = sess.run(dot_op)
print(result)
sess.close()

# method 2 use version
with tf.Session() as sess:
    result_ = sess.run(dot_op)
    print(result_)


if __name__=="__main__":
    pass
    
    