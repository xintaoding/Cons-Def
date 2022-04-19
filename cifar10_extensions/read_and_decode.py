#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 09:41:15 2020

@author: lab
"""
import numpy as np
import tensorflow as tf
#from cleverhans.augmentation import random_horizontal_flip, random_shift

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# open tfrecorder reader
def read_and_decode(filename_queue, nb_epochs, batch_size=128, image_W=32, image_H=32, image_C=3):
#    filename_queue=tf.train.string_input_producer(filename_queue,num_epochs=2,shuffle=True)
    filename_queue=tf.train.string_input_producer(filename_queue, shuffle=True,num_epochs=nb_epochs)
    reader = tf.TFRecordReader()
 
# read file
    _, serialized_example = reader.read(filename_queue)
 
# read data
    features = tf.parse_single_example(serialized_example,
       features={'image_raw': tf.FixedLenFeature([],tf.string),
#   features={'img': tf.FixedLenFeature([3072],tf.float32),
             'height': tf.FixedLenFeature([], tf.int64),
             'width': tf.FixedLenFeature([], tf.int64),
             'depth': tf.FixedLenFeature([], tf.int64),
             'label':tf.FixedLenFeature([],tf.int64)})
 
    img=tf.decode_raw(features['image_raw'],tf.uint8)
    tf_height = features['height']
    tf_width = features['width']
    tf_depth = features['depth']
    
    img=tf.cast(img,tf.float32)*(1./255)
#img=features['img']
    img=tf.reshape(img,[image_W,image_H,image_C])
    label=tf.cast(features['label'],tf.int64)

#    padding = 4
#    img = tf.pad(img,[[padding,padding],[padding,padding],[0,0]], mode='REFLECT')
    img = tf.pad(img,[[4,4],[4,4],[0,0]], mode='REFLECT')
    img=tf.random_crop(img,[image_H,image_W,3])
#    img = random_shift(img)
    img=tf.image.random_flip_left_right(img)
    min_after_dequeue=500000
    
    capacity=min_after_dequeue+3*batch_size
    img, label = tf.train.shuffle_batch([img,label],batch_size=batch_size,num_threads=3, capacity=capacity,min_after_dequeue=min_after_dequeue)
#    img, label = tf.train.batch([img,label],batch_size=bs,capacity=capacity)
    return img, label
