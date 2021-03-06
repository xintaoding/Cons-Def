"""
This file used to train a model using augmented images with TensorFlow.
The augmentations should be prepared for this implementation.
Revised from the work of PanJinquan: https://github.com/PanJinquan/tensorflow_models_learning
Xintao Ding
School of Computer and Information, Anhui Normal University
xintaoding@163.com
"""#coding=utf-8
import tensorflow as tf 
import numpy as np 
#import pdb
import os
from datetime import datetime
import slim.nets.resnet_v1 as resnet_v1
from create_tf_record import get_example_nums,read_records,get_batch_images
import tensorflow.contrib.slim as slim

labels_nums = 10  #  the number of labels
batch_size = 50  #depend your GPU memory, a larger
resize_height = 299  # ImageNet size
resize_width = 299
net_height = 224# ResNet size
net_width = 224

depths = 3
data_shape = [batch_size, resize_height, resize_width, depths]

#input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')
input_images = tf.placeholder(dtype=tf.float32, shape=[None, net_height, net_width, depths], name='input')
input_labels = tf.placeholder(dtype=tf.int32, shape=[None, labels_nums], name='label')

is_training = tf.placeholder(tf.bool, name='is_training')

def net_evaluation(sess,loss,accuracy,val_images_batch,val_labels_batch,val_nums):
    val_max_steps = int(val_nums / batch_size)
    val_losses = []
    val_accs = []
    for _ in range(val_max_steps):
#        val_images_batch = tf.random_crop(val_images_batch,[batch_size,224,224,3])
#        val_images_batch = tf.image.random_flip_left_right(val_images_batch)
        val_x, val_y = sess.run([val_images_batch, val_labels_batch])
        val_loss,val_acc = sess.run([loss,accuracy], feed_dict={input_images: val_x, input_labels: val_y, is_training: False})
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    mean_loss = np.array(val_losses, dtype=np.float32).mean()
    mean_acc = np.array(val_accs, dtype=np.float32).mean()
    return mean_loss, mean_acc

def step_train(train_op,loss,accuracy,
               train_images_batch,train_labels_batch,train_nums,train_log_step,
               val_images_batch,val_labels_batch,val_nums,val_log_step,
               snapshot_prefix,snapshot):
    '''
    ????????????????????????
    :param train_op: ??????op
    :param loss:     loss??????
    :param accuracy: ???????????????
    :param train_images_batch: ??????images??????
    :param train_labels_batch: ??????labels??????
    :param train_nums:         ???????????????
    :param train_log_step:   ??????log????????????
    :param val_images_batch: ??????images??????
    :param val_labels_batch: ??????labels??????
    :param val_nums:         ???????????????
    :param val_log_step:     ??????log????????????
    :param snapshot_prefix: ?????????????????????
    :param snapshot:        ??????????????????
    :return: None
    '''
    saver = tf.train.Saver(max_to_keep=100)
    max_acc = 0.0
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(max_steps + 1):
#            train_images_batch = tf.random_crop(train_images_batch,[batch_size,224,224,3])
#            train_images_batch = tf.image.random_flip_left_right(train_images_batch)
            batch_input_images, batch_input_labels = sess.run([train_images_batch, train_labels_batch])
            _, train_loss = sess.run([train_op, loss], feed_dict={input_images: batch_input_images,
                                                                  input_labels: batch_input_labels,
             #                                                     keep_prob: 0.8, is_training: True})
                                                           is_training: True})
#            print("max_steps:{},train_op:{}".format(max_steps,train_op))
            # train??????(?????????????????????????????????batch)
            if i % train_log_step == 0:
                train_acc = sess.run(accuracy, feed_dict={input_images: batch_input_images,
                                                          input_labels: batch_input_labels,
#                                                          keep_prob: 1.0, is_training: False})
                                                          is_training: False})
                print("%s: Step [%d]  train Loss : %f, training accuracy :  %g" % (
                datetime.now(), i, train_loss, train_acc))

            # val??????(????????????val??????)
            if i % val_log_step == 0:
                mean_loss, mean_acc = net_evaluation(sess, loss, accuracy, val_images_batch, val_labels_batch, val_nums)
                print("%s: Step [%d]  val Loss : %f, val accuracy :  %g" % (datetime.now(), i, mean_loss, mean_acc))

            # ????????????:?????????snapshot?????????????????????????????????
            if (i % snapshot == 0 and i > 0) or i == max_steps:
                print('-----save:{}-{}'.format(snapshot_prefix, i))
                saver.save(sess, snapshot_prefix, global_step=i)
            # ??????val????????????????????????
            if mean_acc > max_acc and mean_acc > 0.7:
                max_acc = mean_acc
                path = os.path.dirname(snapshot_prefix)
                best_models = os.path.join(path, 'resnet50_best_models_{}_{:.4f}.ckpt'.format(i, max_acc))
                print('------save:{}'.format(best_models))
                saver.save(sess, best_models)

        coord.request_stop()
        coord.join(threads)

def train(train_record_file,
          train_log_step,
          train_param,
          val_record_file,
          val_log_step,
          labels_nums,
          data_shape,
          snapshot,
          snapshot_prefix):
    '''
    :param train_record_file: ?????????tfrecord??????
    :param train_log_step: ??????????????????log????????????
    :param train_param: train??????
    :param val_record_file: ?????????tfrecord??????
    :param val_log_step: ??????????????????log????????????
    :param val_param: val??????
    :param labels_nums: labels???
    :param data_shape: ????????????shape
    :param snapshot: ??????????????????
    :param snapshot_prefix: ??????????????????????????????
    :return:
    '''
    [base_lr,max_steps]=train_param
    [batch_size,resize_height,resize_width,depths]=data_shape

    # ?????????????????????????????????
    if len(train_record_file)>1:
        train_nums=845000    
    else:
        train_nums=get_example_nums(train_record_file[0])
    for i in range(len(val_record_file)):
        val_nums=get_example_nums(val_record_file[i])
    print('train nums:%d,val nums:%d'%(train_nums,val_nums))

    # ???record??????????????????labels??????
    # train??????,????????????????????????????????????shuffle=True
    train_images, train_labels = read_records(train_record_file, resize_height, resize_width, 
    #                                          net_height=net_height, net_width=net_width, 
    #                                          type='normalization',crop_flip=True)
                                              type='centralization',crop=True, flip=True)
    train_images_batch, train_labels_batch = get_batch_images(train_images, train_labels,
                                                              batch_size=batch_size, labels_nums=labels_nums,
                                                              one_hot=True, shuffle=True)
    train_images_batch = tf.image.resize_images(train_images_batch,size=(net_height, net_width))
    # val??????,???????????????????????????????????????
    val_images, val_labels = read_records(val_record_file, resize_height=net_height, resize_width=net_width, 
    #                                      net_height=net_height, net_width=net_width,
    #                                     type='normalization',crop_flip=True)
                                          type='centralization')

    val_images_batch, val_labels_batch = get_batch_images(val_images, val_labels,
                                                          batch_size=batch_size, labels_nums=labels_nums,
                                                          one_hot=True, shuffle=False)
#    val_images_batch = tf.image.resize_images(val_images_batch,size=(net_height, net_width))

    # Define the model:
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    #    out, end_points = resnet_v1.resnet_v1_101(inputs=input_images, num_classes=labels_nums, is_training=is_training,global_pool=True)
    #    out, end_points = resnet_v1.resnet_v1_101(inputs=input_images, num_classes=labels_nums, is_training=is_training,global_pool=True)
        out, end_points = resnet_v1.resnet_v1_50(inputs=input_images, num_classes=labels_nums, is_training=is_training,global_pool=True)
    # with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
    #     out, end_points = mobilenet_v1.mobilenet_v1(inputs=input_images, num_classes=labels_nums,
    #                                                 dropout_keep_prob=keep_prob, is_training=is_training,
    #                                                 global_pool=True)

        # Specify the loss function: tf.losses?????????loss???????????????????????????loss??????,?????????add_loss()???
    tf.losses.softmax_cross_entropy(onehot_labels=input_labels, logits=out)
    # slim.losses.add_loss(my_loss)
    loss = tf.losses.get_total_loss(add_regularization_losses=True)

    # Specify the optimization scheme:

    # ????????????????????????, ????????????????????????`batch_norm`??????,????????????????????????`average`???`variance`??????,
    # ???????????????????????????????????????????????????, ??????????????????????????????????????????
    # ??????`tf.get_collection`???????????????????????????`op`
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # ??????`tensorflow`????????????, ?????????????????????, ???????????????
    with tf.control_dependencies(update_ops):
 #       print("update_ops:{}".format(update_ops))
        # create_train_op that ensures that when we evaluate it to get the loss,
        # the update_ops are done and the gradient updates are computed.
        # train_op = tf.train.MomentumOptimizer(learning_rate=base_lr, momentum=0.9).minimize(loss)
        train_op = tf.train.AdadeltaOptimizer(learning_rate=base_lr).minimize(loss)


    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(input_labels, 1)), tf.float32))
    # ??????????????????
    step_train(train_op=train_op, loss=loss, accuracy=accuracy,
               train_images_batch=train_images_batch,
               train_labels_batch=train_labels_batch,
               train_nums=train_nums,
               train_log_step=train_log_step,
               val_images_batch=val_images_batch,
               val_labels_batch=val_labels_batch,
               val_nums=val_nums,
               val_log_step=val_log_step,
               snapshot_prefix=snapshot_prefix,
               snapshot=snapshot)

if __name__ == '__main__':
#    train_record_file=['dataset/record/train299_first10cls.tfrecords']#includes 13k examples
    train_record_file=['data/caffe_ilsvrc12_record/train299.tfrecords_seg0',
                       'data/caffe_ilsvrc12_record/train299.tfrecords_seg1',
                       'data/caffe_ilsvrc12_record/train299.tfrecords_seg2',
                       'data/caffe_ilsvrc12_record/train299.tfrecords_seg3',
                       'data/caffe_ilsvrc12_record/train299.tfrecords_seg4',
                       'data/caffe_ilsvrc12_record/train299.tfrecords_seg5',
                       'data/caffe_ilsvrc12_record/train299.tfrecords_seg6',
                       'data/caffe_ilsvrc12_record/train299.tfrecords_seg7',
                       'data/caffe_ilsvrc12_record/train299.tfrecords_seg8',
                       'data/caffe_ilsvrc12_record/train299.tfrecords_seg9',
                       'data/caffe_ilsvrc12_record/train299.tfrecords_seg10',
                       'data/caffe_ilsvrc12_record/train299.tfrecords_seg11',
                       'data/caffe_ilsvrc12_record/train299.tfrecords_seg12']
    val_record_file=['data/caffe_ilsvrc12_record/val224.tfrecords']

    train_log_step=200
    base_lr = 0.1#0.001  learning rate
    max_steps = 1700000#200000  # train 100 epochs, approximately 1700k iterations, 1700000=13000*65*100/50,max_steps=N_samples*epoches/batch_size
    train_param=[base_lr,max_steps]

    val_log_step=500
    snapshot=20000
    snapshot_prefix='models/caffe_ilsvrc12/res50_fisrt10cls_model.ckpt'
    train(train_record_file=train_record_file,
          train_log_step=train_log_step,
          train_param=train_param,
          val_record_file=val_record_file,
          val_log_step=val_log_step,
          labels_nums=labels_nums,
          data_shape=data_shape,
          snapshot=snapshot,
          snapshot_prefix=snapshot_prefix)
