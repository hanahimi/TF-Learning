import tensorflow as tf


class Donkey(object):
    @staticmethod
    def _preprocess(image):
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        
#         image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.multiply(tf.subtract(image, 0.5), 2)
        image = tf.reshape(image, [64, 64, 3])
#         image = tf.image.random_contrast(image,lower=0.5,upper=1.5)
        image = tf.random_crop(image, [60, 60, 3])
        image = tf.image.resize_images(image, [64,64])
        return image

    @staticmethod
    def _read_and_decode(filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'length': tf.FixedLenFeature([], tf.int64),
                'digits': tf.FixedLenFeature([5], tf.int64)
            })

        image = Donkey._preprocess(tf.decode_raw(features['image'], tf.uint8))
        length = tf.cast(features['length'], tf.int32)
        digits = tf.cast(features['digits'], tf.int32)
        return image, length, digits

    @staticmethod
    def build_batch(path_to_tfrecords_file, num_examples, batch_size, shuffled):
        assert tf.gfile.Exists(path_to_tfrecords_file), '%s not found' % path_to_tfrecords_file

        filename_queue = tf.train.string_input_producer([path_to_tfrecords_file], num_epochs=None)
        image, length, digits = Donkey._read_and_decode(filename_queue)

        min_queue_examples = int(0.4 * num_examples)
        if shuffled:
            image_batch, length_batch, digits_batch = tf.train.shuffle_batch([image, length, digits],
                                                                             batch_size=batch_size,
                                                                             num_threads=2,
                                                                             capacity=min_queue_examples + 3 * batch_size,
                                                                             min_after_dequeue=min_queue_examples)
        else:
            image_batch, length_batch, digits_batch = tf.train.batch([image, length, digits],
                                                                     batch_size=batch_size,
                                                                     num_threads=2,
                                                                     capacity=min_queue_examples + 3 * batch_size)
        return image_batch, length_batch, digits_batch


# import os
# from util.meta import Meta
# tf.app.flags.DEFINE_string('data_dir', r'D:\conda_env\workspace\py\TF-Learning\src\e2e_ocrnet\dataset\character_num', 'Directory to read TFRecords files')
# FLAGS = tf.app.flags.FLAGS
# 
# def main(_):
#     path_to_train_tfrecords_file = os.path.join(FLAGS.data_dir, 'train.tfrecords')
#     path_to_val_tfrecords_file = os.path.join(FLAGS.data_dir, 'val.tfrecords')
#     path_to_tfrecords_meta_file = os.path.join(FLAGS.data_dir, 'meta.json')
#     
#     meta = Meta()
#     meta.load(path_to_tfrecords_meta_file)
#     
#     with tf.Graph().as_default():
#         image_batch, length_batch, chars_batch = Donkey.build_batch(path_to_train_tfrecords_file,
#                                                                      num_examples=meta.num_train_examples,
#                                                                      batch_size=32,
#                                                                      shuffled=True)
#         with tf.Session() as sess:
#             sess.run(tf.global_variables_initializer())
#             
#             
#             
# if __name__ == '__main__':
#     tf.app.run(main=main)
    
    
    