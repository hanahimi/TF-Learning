#-*-coding:UTF-8-*-
'''
Created on 2016-8-9

@author: hanahimi
'''
import tensorflow as tf
import numpy as np
import cv2
import time
from dataio import *
# from datatype import DataPackage
from prep_2_make_datapkl import DataPackage, ImgDataset

learning_rate = 0.001
training_iters = 1000
batch_size = 256
val_step = 20

def train_valid_test(x, keep_prob, model_pred, model_name, pkl_name, input_size, isTrain=False):
    dataset = load_pickle(r'dataset/%s.pkl' % pkl_name)
    print("dataset load success")
    n_classes = dataset.class_num
    y_ = tf.placeholder("float",shape = [None, n_classes])
    
    pred = model_pred
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y_))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # 瀹氫箟缃戠粶娴嬭瘯
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.scalar_summary('accuracy', accuracy)

    saver = tf.train.Saver()
    is_restore = True
    
    init = tf.initialize_all_variables()
    
    with tf.Session()  as sess:
        log = open(("./log/%s_%d.log" % (model_name,input_size)), 'a')
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        log.write("start time:\n%s\n" % timestamp)
        log.write("training:\n")
        
        if is_restore==True:
            try:
                saver.restore(sess, ("./save_ckpt/%s_%d.ckpt" % (model_name,input_size)))
                print("Model restored")
            except:
                print("Model restored error, turn to init")
                sess.run(init)
                print("Model inited")
        else:
            sess.run(init)
            print("Model inited")
        if isTrain:
            step = 1
            while step * batch_size < training_iters:
                batch_xs, batch_ys = dataset.train.next_batch(batch_size)
                tic = time.time()
                sess.run(optimizer,
                         feed_dict={x:batch_xs, y_:batch_ys,keep_prob:0.8})
                toc = time.time() - tic
                
                if step % val_step == 0:
#                     batch_xs,batch_ys = dataset.val.all_batches()
                    acc = sess.run(
                        accuracy,
                        feed_dict={x:batch_xs, y_:batch_ys,keep_prob:1})
                    loss = sess.run(
                        cost,
                        feed_dict={x:batch_xs, y_:batch_ys,keep_prob:1})
                    iter_rs = ("Iter:%d Minibatch Loss=%.5f Train Accuracy=%.5f training time=%.3f" % 
                                (step*batch_size, loss, acc, toc))
                    print(iter_rs)
                    save_path = saver.save(sess, ("./save_ckpt/%s_%d.ckpt" % (model_name,input_size)))
                    print("Model saved in file: ", save_path)
                    log.write("%s\n" % iter_rs)

                step += 1
                print("step: %d" % step)
            print("Optimization Finished!")
            save_path = saver.save(sess, ("./save_ckpt/%s_%d.ckpt" % (model_name,input_size)))
            print("Model saved in file: ", save_path)
        
            log.write("end training:\n")
            timestamp = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
            log.write("current time:%s\n" % timestamp)
        log.write("start testing:\n")

        
        tic = time.time()
        batch_xs, batch_ys = dataset.test.all_batches()
        acc_rate = sess.run(accuracy, 
                            feed_dict={x: batch_xs, 
                                       y_: batch_ys, 
                                       keep_prob: 1.})

        print("Testing Accuracy:", acc_rate)
        toc = time.time() - tic
        print(toc,toc/len(dataset.test.labels))
        
        log.write("end testing:\n")
        log.write("Testing Accuracy:%f, total times:%f, avg time:%f\n" % 
                  (acc_rate,toc,toc/len(dataset.test.labels)))
        
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        log.write("current time:%s\n" % timestamp)
        log.close()

if __name__=="__main__":
    pass

    
    