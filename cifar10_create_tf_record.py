# -*-coding: utf-8 -*-
"""
    @Project: create_tfrecord
    @File   : create_tfrecord.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-07-27 17:19:54
    @desc   : ��ͼƬ���ݱ���Ϊ����tfrecord�ļ�
    
    Revised by Xintao Ding
    Build tfrecord files for the training and testing of cifar10 dataset
    If aug is set 'True', the training examples are augmented to 10 tfrecords segments: _seg0, _seg1, ..., _seg9
    Else the training examples are packed without augmentation
    The testing examples are packed in a tfrecord file without augmentation.
    Our Cons-Def method augment them in different size for different networks, such as vgg16, resnet50
"""

##########################################################################

import tensorflow as tf
import numpy as np
from cleverhans.dataset import CIFAR10
import matplotlib.pyplot as plt
from PIL import Image
from cleverhans.data_extenv2 import data_exten
from cifar10_extensions.read_and_decode import read_and_decode

##########################################################################
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
# �����ַ����͵�����
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
# ����ʵ���͵�����
def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def get_example_nums(tf_records_filenames):
    '''
    ͳ��tf_recordsͼ��ĸ���(example)����
    :param tf_records_filenames: tf_records�ļ�·��
    :return:
    '''
    nums= 0
    for record in tf.python_io.tf_record_iterator(tf_records_filenames):
        nums += 1
    return nums
  
def read_records(filename,image_W=32, image_H=32, image_C=3, type=None, crop_flip=False):
    '''
    ����record�ļ�:Դ�ļ���ͼ��������RGB,uint8,[0,255],һ����Ϊѵ������ʱ,��Ҫ��һ����[0,1]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param type:ѡ��ͼ�����ݵķ�������
         None:Ĭ�Ͻ�uint8-[0,255]תΪfloat32-[0,255]
         normalization:��һ��float32-[0,1]
         centralization:��һ��float32-[0,1],�ټ���ֵ���Ļ�
    :return:
    '''

    # �����ļ�����,���޶�ȡ������
#    filename_queue = tf.train.string_input_producer([filename])
    filename_queue = tf.train.string_input_producer(filename, shuffle=True)
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader���ļ������ж���һ�����л�������
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # �������Ż�������
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    tf_image = tf.decode_raw(features['image_raw'], tf.uint8)#���ͼ��ԭʼ������

    tf_height = features['height']
    tf_width = features['width']
    tf_depth = features['depth']
    tf_label = tf.cast(features['label'], tf.int32)
    # PS:�ָ�ԭʼͼ������,reshape�Ĵ�С�����뱣��֮ǰ��ͼ��shapeһ��,�������
    # tf_image=tf.reshape(tf_image, [-1])    # ת��Ϊ������
    tf_image=tf.reshape(tf_image, [image_H, image_W, image_C]) # ����ͼ���ά��[resize_height, resize_width, 3]

    # �ָ����ݺ�,�ſ��Զ�ͼ�����resize_images:����uint->���float32
    # tf_image=tf.image.resize_images(tf_image,[224, 224])

    # �洢��ͼ������Ϊuint8,tensorflowѵ��ʱ���ݱ�����tf.float32
    if type is None:
        tf_image = tf.cast(tf_image, tf.float32)
    elif type=='normalization':# [1]����Ҫ��һ����ʹ��:
        # ��������������uint8,�Ż��һ��[0,255]
        # tf_image = tf.image.convert_image_dtype(tf_image, tf.float32)
        tf_image = tf.cast(tf_image, tf.float32) * (1. / 255.0)  # ��һ��
    elif type=='centralization':
        # ����Ҫ��һ��,�����Ļ�,�����ֵΪ0.5,��ʹ��:
        tf_image = tf.cast(tf_image, tf.float32) * (1. / 255) - 0.5 #���Ļ�

    if crop_flip:
        pad=(4, 4)
        assert tf_image.get_shape().ndims == 3
        xp = tf.pad(tf_image, [[pad[0], pad[0]], [pad[1], pad[1]], [0, 0]], mode='REFLECT')
        tf_image = tf.random_crop(xp, tf.shape(tf_image))
        tf_image = tf.image.random_flip_left_right(tf_image)

    # �����������ͼ��ͱ�ǩ
    # return tf_image, tf_height,tf_width,tf_depth,tf_label
    return tf_image,tf_label
  
def get_batch_images(images,labels,batch_size,labels_nums,one_hot=False,shuffle=False,num_threads=2):
    '''
    :param images:ͼ��
    :param labels:��ǩ
    :param batch_size:
    :param labels_nums:��ǩ����
    :param one_hot:�Ƿ�labelsתΪone_hot����ʽ
    :param shuffle:�Ƿ����˳��,һ��trainʱshuffle=True,��֤ʱshuffle=False
    :return:����batch��images��labels
    '''
    min_after_dequeue = 100000#revised by Ding ====================
    capacity = min_after_dequeue + 3 * batch_size  # ��֤capacity�������min_after_dequeue����ֵ
    if shuffle:
        images_batch, labels_batch = tf.train.shuffle_batch([images,labels],
                                                                    batch_size=batch_size,
                                                                    capacity=capacity,
                                                                    min_after_dequeue=min_after_dequeue,
                                                                    num_threads=num_threads#revised by Ding, 
                                                                    )
    else:
        images_batch, labels_batch = tf.train.batch([images,labels],
                                                        batch_size=batch_size,
                                                        capacity=capacity,
                                                        num_threads=num_threads)
    if one_hot:
        labels_batch = tf.one_hot(labels_batch, labels_nums, 1, 0)
    return images_batch,labels_batch
    
def show_image(title,image):
    '''
    ��ʾͼƬ
    :param title: ͼ�����
    :param image: ͼ�������
    :return:
    '''
    # plt.figure("show_image")
    # print(image.dtype)
    plt.imshow(image)
    plt.axis('on')    # �ص�������Ϊ off
    plt.title(title)  # ͼ����Ŀ
    plt.show()    
  
  


def disp_records(record_file,resize_height, resize_width,show_nums=1):
    '''
    ����record�ļ�������ʾshow_nums��ͼƬ����Ҫ������֤����record�ļ��Ƿ�ɹ�
    :param tfrecord_file: record�ļ�·��
    :return:
    '''
    # ��ȡrecord����
    tf_image, tf_label = read_records(record_file)#),type='normalization')
    tf_image = tf.image.resize_images(tf_image,size=(128, 128))
    # ��ʾǰ1��ͼƬ
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(show_nums):
            image,label = sess.run([tf_image,tf_label])  # �ڻỰ��ȡ��image��label
            # image = tf_image.eval()
            # ֱ�Ӵ�record������image��һ������,��Ҫreshape��ʾ
            # image = image.reshape([height,width,depth])
            image=image.astype(np.uint8)
            print('shape:{},tpye:{},labels:{}'.format(image.shape,image.dtype,label))
            # pilimg = Image.fromarray(np.asarray(image_eval_reshape))
            # pilimg.show()
            show_image("image:%d"%(label),image)
        coord.request_stop()
        coord.join(threads)        

if __name__ == '__main__':
#  train_record_output = 'cifar10_extensions/cifar10_train.tfrecords_seg'
  ''' 
  ʵ�ֽ�ͼ��ԭʼ����,label,��,�����Ϣ����Ϊrecord�ļ�
  ע��:��ȡ��ͼ������Ĭ����uint8,��תΪtf���ַ�����BytesList����,��������Ҫ������Ҫת������
  :param resize_height:
  :param resize_width:
  PS:��resize_height����resize_width=0��,��ִ��resize
  :param shuffle:�Ƿ����˳��
  :param log:log��Ϣ��ӡ���
  '''
  train_end = 50000#the training number of cifar10
  test_end = 10000#testing examples
  data = CIFAR10(train_start=0, train_end=train_end, test_start=0, test_end=test_end)
  dataset_size = data.x_train.shape[0]
  x_train, y_train = data.get_set('train')
  x_test, y_test = data.get_set('test')
  seg_len=5000#divide the tfrecord files in 10 segments, every segment contains 5000 training examples
  n_segs=dataset_size/seg_len
  n_segs=np.int32(n_segs)
  aug=True#branch switch to pack augmented training examples or training examples
#  train_record_output = ['cifar10_extensions/cifar10_test.tfrecords_seg']#test file name
#  disp_records(train_record_output,32, 32)

  if aug:
    train_record_output = 'cifar10_extensions/cifar10_train_aug64.tfrecords'
    for i in range(n_segs):
      writer = tf.python_io.TFRecordWriter(train_record_output+'_seg'+str(i))
      x_traini, y_traini = data_exten(x_train[i*seg_len:(i+1)*seg_len,:,:,:], y_train[i*seg_len:(i+1)*seg_len,:], seg_len, base_range=4)
      for j in range(len(x_traini)):
        image = x_traini[j,:,:,:]*255
        image=image.astype(np.uint8)
        image_raw = image.tostring()
        label=y_traini[j]
        label=np.argmax(label)
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'height': _int64_feature(image.shape[0]),
            'width': _int64_feature(image.shape[1]),
            'depth': _int64_feature(image.shape[2]),
            'label': _int64_feature(label)
        }))
        writer.write(example.SerializeToString())
      writer.close()

  else: 
    val_record_output = 'cifar10_extensions/cifar10_train.tfrecords'
    writer = tf.python_io.TFRecordWriter(val_record_output+'_seg')
    for i in range(train_end):
      image = x_train[i,:,:,:]*255
      image=image.astype(np.uint8)
      image_raw = image.tostring()
      label=y_train[i]
      label=np.argmax(label)
      example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'height': _int64_feature(image.shape[0]),
            'width': _int64_feature(image.shape[1]),
            'depth': _int64_feature(image.shape[2]),
            'label': _int64_feature(label)
      }))
      writer.write(example.SerializeToString())
#    if i%500==499 and i>0:
    writer.close()

  val_record_output = 'cifar10_extensions/cifar10_test.tfrecords'
  writer = tf.python_io.TFRecordWriter(val_record_output+'_seg')
  for i in range(test_end):
      image = x_test[i,:,:,:]*255
      image=image.astype(np.uint8)
      image_raw = image.tostring()
      label=y_test[i]
      label=np.argmax(label)
      example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'height': _int64_feature(image.shape[0]),
            'width': _int64_feature(image.shape[1]),
            'depth': _int64_feature(image.shape[2]),
            'label': _int64_feature(label)
      }))
      writer.write(example.SerializeToString())
#    if i%500==499 and i>0:
  writer.close()
    
    
  train_record_output = ['cifar10_extensions/cifar10_test.tfrecords_seg']
  disp_records(train_record_output,32, 32)





