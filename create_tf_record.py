# -*-coding: utf-8 -*-
"""
    @Project: create_tfrecord
    @File   : create_tfrecord.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-07-27 17:19:54
    @desc   : ��ͼƬ���ݱ���Ϊ����tfrecord�ļ�
"""

##########################################################################

import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random
from PIL import Image


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
#        print("nums:{}".format(nums))
    return nums

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

def load_labels_file(filename,labels_num=1,shuffle=False):
    '''
    ��ͼtxt�ļ����ļ���ÿ��Ϊһ��ͼƬ��Ϣ�����Կո������ͼ��·�� ��ǩ1 ��ǩ2���磺test_image/1.jpg 0 2
    :param filename:
    :param labels_num :labels����
    :param shuffle :�Ƿ����˳��
    :return:images type->list
    :return:labels type->list
    '''
    images=[]
    labels=[]
    with open(filename) as f:
        lines_list=f.readlines()
        if shuffle:
            random.shuffle(lines_list)

        for lines in lines_list:
            line=lines.rstrip().split(' ')
            label=[]
            for i in range(labels_num):
                label.append(int(line[i+1]))
            images.append(line[0])
            labels.append(label)
    return images,labels

def read_image(filename, resize_height, resize_width,normalization=False):
    '''
    ��ȡͼƬ����,Ĭ�Ϸ��ص���uint8,[0,255]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param normalization:�Ƿ��һ����[0.,1.0]
    :return: ���ص�ͼƬ����
    '''

    bgr_image = cv2.imread(filename)
    if len(bgr_image.shape)==2:#���ǻҶ�ͼ��תΪ��ͨ��
        print("Warning:gray image",filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)#��BGRתΪRGB
    # show_image(filename,rgb_image)
    # rgb_image=Image.open(filename)
    if resize_height>0 and resize_width>0:
        rgb_image=cv2.resize(rgb_image,(resize_width,resize_height))
    rgb_image=np.asanyarray(rgb_image)
    if normalization:
        # ����д��:rgb_image=rgb_image/255
        rgb_image=rgb_image/255.0
    # show_image("src resize image",image)
    return rgb_image


def get_batch_images(images,labels,batch_size,labels_nums,one_hot=False,shuffle=False,num_threads=3):
    '''
    :param images:ͼ��
    :param labels:��ǩ
    :param batch_size:
    :param labels_nums:��ǩ����
    :param one_hot:�Ƿ�labelsתΪone_hot����ʽ
    :param shuffle:�Ƿ����˳��,һ��trainʱshuffle=True,��֤ʱshuffle=False
    :return:����batch��images��labels
    '''
    min_after_dequeue = 3200#revised by Ding ====================
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

def read_records(filename,resize_height, resize_width,type=None,padding=False, crop=False,flip=False):
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
    tf_image=tf.reshape(tf_image, [resize_height, resize_width, 3]) # ����ͼ���ά��

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

    if padding:
        pad=(30, 30)
        assert tf_image.get_shape().ndims == 3
#        xp = tf.pad(tf_image, [[pad[0], pad[0]], [pad[1], pad[1]], [0, 0]], mode='REFLECT')
        tf_image = tf.pad(tf_image, [[pad[0], pad[0]], [pad[1], pad[1]], [0, 0]], mode='REFLECT')
    if crop:
        tf_image = tf.random_crop(tf_image, [resize_height,resize_height,3])
    if flip:
        tf_image = tf.image.random_flip_left_right(tf_image)
    # �����������ͼ��ͱ�ǩ
    # return tf_image, tf_height,tf_width,tf_depth,tf_label
    return tf_image,tf_label


def create_records(image_dir,file, output_record_dir, resize_height, resize_width,shuffle,log=5):
    '''
    ʵ�ֽ�ͼ��ԭʼ����,label,��,�����Ϣ����Ϊrecord�ļ�
    ע��:��ȡ��ͼ������Ĭ����uint8,��תΪtf���ַ�����BytesList����,��������Ҫ������Ҫת������
    :param image_dir:ԭʼͼ���Ŀ¼
    :param file:���뱣��ͼƬ��Ϣ��txt�ļ�(image_dir+file����ͼƬ��·��)
    :param output_record_dir:����record�ļ���·��
    :param resize_height:
    :param resize_width:
    PS:��resize_height����resize_width=0��,��ִ��resize
    :param shuffle:�Ƿ����˳��
    :param log:log��Ϣ��ӡ���
    '''
    # �����ļ�,����ȡһ��label
    images_list, labels_list=load_labels_file(file,1,shuffle)

    writer = tf.python_io.TFRecordWriter(output_record_dir)
    for i, [image_name, labels] in enumerate(zip(images_list, labels_list)):
        image_path=os.path.join(image_dir,images_list[i])
        if not os.path.exists(image_path):
            print('Err:no image',image_path)
            continue
        image = read_image(image_path, resize_height, resize_width)
        image_raw = image.tostring()
        if i%log==0 or i==len(images_list)-1:
            print('------------processing:%d-th------------' % (i))
            print('current image_path=%s' % (image_path),'shape:{}'.format(image.shape),'labels:{}'.format(labels))
        # ���������һ��label,��label�ʵ�����"'label': _int64_feature(label)"��
        label=labels[0]
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'height': _int64_feature(image.shape[0]),
            'width': _int64_feature(image.shape[1]),
            'depth': _int64_feature(image.shape[2]),
            'label': _int64_feature(label)
        }))
        writer.write(example.SerializeToString())
    writer.close()

def disp_records(record_file,resize_height, resize_width,show_nums=65):
    '''
    ����record�ļ�������ʾshow_nums��ͼƬ����Ҫ������֤����record�ļ��Ƿ�ɹ�
    :param tfrecord_file: record�ļ�·��
    :return:
    '''
    # ��ȡrecord����
    tf_image, tf_label = read_records(record_file,resize_height,resize_width,type='normalization')
    # ��ʾǰ4��ͼƬ
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(show_nums):
            image,label = sess.run([tf_image,tf_label])  # �ڻỰ��ȡ��image��label
            # image = tf_image.eval()
            # ֱ�Ӵ�record������image��һ������,��Ҫreshape��ʾ
            # image = image.reshape([height,width,depth])
            print('shape:{},tpye:{},labels:{}'.format(image.shape,image.dtype,label))
            # pilimg = Image.fromarray(np.asarray(image_eval_reshape))
            # pilimg.show()
        image=image*255.0
        image=image.astype(np.uint8)
        show_image("image:%d"%(label),image)
        coord.request_stop()
        coord.join(threads)


def batch_test(record_file,resize_height, resize_width):
    '''
    :param record_file: record�ļ�·��
    :param resize_height:
    :param resize_width:
    :return:
    :PS:image_batch, label_batchһ����Ϊ���������
    '''
    # ��ȡrecord����
    tf_image,tf_label = read_records(record_file,resize_height,resize_width,type='normalization')
    image_batch, label_batch= get_batch_images(tf_image,tf_label,batch_size=4,labels_nums=5,one_hot=False,shuffle=False)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:  # ��ʼһ���Ự
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(4):
            # �ڻỰ��ȡ��images��labels
            images, labels = sess.run([image_batch, label_batch])
            # �������ʾÿ��batch���һ��ͼƬ
            show_image("image", images[0, :, :, :])
            print('shape:{},tpye:{},labels:{}'.format(images.shape,images.dtype,labels))

        # ֹͣ�����߳�
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    # ��������

    resize_height = 299  # ָ���洢ͼƬ�߶�
    resize_width = 299  # ָ���洢ͼƬ���
    shuffle=True
    log=5
    # ����train.record�ļ�
    image_dir='caffe_ilsvrc12/train'
    train_labels = 'caffe_ilsvrc12/train.txt'  # ͼƬ·��
    train_record_output = 'data/caffe_ilsvrc12_record/train{}.tfrecords_seg0'.format(resize_height)
#    create_records(image_dir,train_labels, train_record_output, resize_height, resize_width,shuffle,log)
#    train_nums=get_example_nums(train_record_output)
#    print("save train example nums={}".format(train_nums))

    # ����val.record�ļ�
    image_dir='caffe_ilsvrc12/val'
    val_labels = 'caffe_ilsvrc12/val.txt'  # ͼƬ·��
    val_record_output = 'data/caffe_ilsvrc12_record/val{}.tfrecords'.format(resize_height)
#    create_records(image_dir,val_labels, val_record_output, resize_height, resize_width,shuffle,log)
#    val_nums=get_example_nums(val_record_output)
#    print("save val example nums={}".format(val_nums))

    # ������ʾ����
    num=get_example_nums(val_record_output)
    disp_records([val_record_output],resize_height, resize_width)
    batch_test(train_record_output,resize_height, resize_width)

