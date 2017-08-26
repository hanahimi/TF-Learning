#-*-coding:UTF-8-*-
'''
Created on 2017年8月9日-下午5:47:24
author: Gary-W
'''
import tensorflow as tf

# assign init value to var
var = tf.Variable(0)    # our first variable in the "global_variable" set

add_op = tf.add(var, 1, "increase")

# in tf 1.1.0, use "assign" to give Variable value
update_op = tf.assign(var, add_op)
update_op = tf.assign_add(var, add_op)

with tf.Session() as sess:
    # once define variables, you have to initialize them by doing this
    sess.run(tf.global_variables_initializer())
    for _ in range(3):
        sess.run(update_op)
        print(sess.run(var))
        print(sess.run(add_op))


if __name__=="__main__":
    pass

