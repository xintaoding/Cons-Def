"""
This file used to train a model using augmented images with TensorFlow.
The augmentations are processed in the function file data_pertprep.py and they are loaded in memory
Revised from the work of PanJinquan: https://github.com/PanJinquan/tensorflow_models_learning
Xintao Ding
School of Computer and Information, Anhui Normal University
xintaoding@163.com
"""#coding=utf-8

import tensorflow as tf 
import numpy as np 
import os
from datetime import datetime
import slim.nets.vgg160 as vgg
from cifar10_create_tf_record import get_example_nums, read_records, get_batch_images
import tensorflow.contrib.slim as slim


labels_nums = 10  #  the number of labels
batch_size = 90  #
resize_height = 32  # Cifar10 size
resize_width = 32  
net_height= 160#the size must be suitable with layer in vgg (net, 4096, [5, 5], padding=fc_conv_padding, scope='fc6'), 224*224 corresponds to [7, 7], 160*160 correpsonds to [5, 5],
net_width = 160#vgg16 size
depths = 3
data_shape = [batch_size, resize_height, resize_width, depths]

#input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')
input_images = tf.placeholder(dtype=tf.float32, shape=[None, net_height, net_width, depths], name='input')
# input_labels = tf.placeholder(dtype=tf.int32, shape=[None], name='label')
input_labels = tf.placeholder(dtype=tf.int32, shape=[None, labels_nums], name='label')

#keep_prob = 0.5
is_training = tf.placeholder(tf.bool, name='is_training')

def net_evaluation(sess,loss,accuracy,val_images_batch,val_labels_batch,val_nums):
    val_max_steps = int(val_nums / batch_size)
    val_losses = []
    val_accs = []
    for _ in range(val_max_steps):
        val_x, val_y = sess.run([val_images_batch, val_labels_batch])
        # print('labels:',val_y)
#        val_loss,val_acc = sess.run([loss,accuracy], feed_dict={input_images: val_x, input_labels: val_y, keep_prob:keep_prob, is_training: False})
        val_loss,val_acc = sess.run([loss,accuracy], feed_dict={input_images: val_x, input_labels: val_y, is_training: False})
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    mean_loss = np.array(val_losses, dtype=np.float32).mean()
    mean_acc = np.array(val_accs, dtype=np.float32).mean()
    return mean_loss, mean_acc


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
    :param train_record_file: 训练的tfrecord文件
    :param train_log_step: 显示训练过程log信息间隔
    :param train_param: train参数
    :param val_record_file: 验证的tfrecord文件
    :param val_log_step: 显示验证过程log信息间隔
    :param val_param: val参数
    :param labels_nums: labels数
    :param data_shape: 输入数据shape
    :param snapshot: 保存模型间隔
    :param snapshot_prefix: 保存模型文件的前缀名
    :return:
    '''
    [base_lr,max_steps]=train_param
    [batch_size,resize_height,resize_width,depths]=data_shape

    # 获得训练和测试的样本数
    train_nums = 0
    for i in range(len(train_record_file)):
        train_nums = train_nums+get_example_nums(train_record_file[i])
    for i in range(len(val_record_file)):
        val_nums=get_example_nums(val_record_file[i])
    print('train nums:%d,val nums:%d'%(train_nums,val_nums))

    # 从record中读取图片和labels数据
    # train数据,训练数据一般要求打乱顺序shuffle=True
    train_images, train_labels = read_records(train_record_file, resize_height, resize_width, depths,
                                              type='normalization', crop_flip=True) 
    train_images_batch, train_labels_batch = get_batch_images(train_images, train_labels,
                                                              batch_size=batch_size, labels_nums=labels_nums,
                                                              one_hot=True, shuffle=True)
    train_images_batch = tf.image.resize_images(train_images_batch,size=(net_height, net_width))
    # val数据,验证数据可以不需要打乱数据
    val_images, val_labels = read_records(val_record_file, resize_height, resize_width, depths, type='normalization')
    val_images_batch, val_labels_batch = get_batch_images(val_images, val_labels,
                                                          batch_size=batch_size, labels_nums=labels_nums,
                                                          one_hot=True, shuffle=False)
    val_images_batch = tf.image.resize_images(val_images_batch,size=(net_height, net_width))
    
    # Define the model:
    with slim.arg_scope(vgg.vgg_arg_scope()):
#        out, end_points = vgg.vgg_16(inputs=input_images, num_classes=labels_nums, keep_prob=keep_prob, is_training=is_training)
        out, end_points = vgg.vgg_16(inputs=input_images, num_classes=labels_nums, is_training=is_training)

    # Specify the loss function: tf.losses定义的loss函数都会自动添加到loss函数,不需要add_loss()了
    tf.losses.softmax_cross_entropy(onehot_labels=input_labels, logits=out)#添加交叉熵损失loss=1.6
    # slim.losses.add_loss(my_loss)
    loss = tf.losses.get_total_loss(add_regularization_losses=True)#添加正则化损失loss=2.2
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(input_labels, 1)), tf.float32))

    # Specify the optimization scheme:
#    optimizer = tf.train.GradientDescentOptimizer(learning_rate=base_lr)
    # create_train_op that ensures that when we evaluate it to get the loss,
    # the update_ops are done and the gradient updates are computed.
#    train_op = slim.learning.create_train_op(total_loss=loss,optimizer=optimizer)
    
    train_op = tf.train.AdadeltaOptimizer(learning_rate=base_lr).minimize(loss)#revised from GradientDescentOptimizer######

#    global_step = tf.Variable(0, trainable=False)
#    learning_rate = tf.train.exponential_decay(0.045, global_step, int(train_nums/batch_size*2),0.94, staircase=True)
#    add_global=global_step.assign_add(1)
    #
#    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
#    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # # train_op = optimizer.minimize(loss, global_step)
#    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#    with tf.control_dependencies(update_ops):
#    train_op = slim.learning.create_train_op(total_loss=loss, optimizer=optimizer,global_step=global_step,clip_gradient_norm=2.)

#    boundaries = [200, 400]
#    lrs = [0.01, 0.001, 0.0001]
#    global_step = tf.Variable(0, trainable=False)
#    add_global=global_step.assign_add(1)
#    learning_rate=tf.train.piecewise_constant(global_step, boundaries=boundaries, values=lrs)
#    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
#    train_op = slim.learning.create_train_op(total_loss=loss, optimizer=optimizer,global_step=global_step,clip_gradient_norm=2.)

    saver = tf.train.Saver(max_to_keep=100)
    max_acc=0.0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(max_steps+1):
            batch_input_images, batch_input_labels = sess.run([train_images_batch, train_labels_batch])
            _, train_loss = sess.run([train_op, loss], feed_dict={input_images:batch_input_images,
                                                                      input_labels:batch_input_labels,
                                                                      is_training:True})
#                                                                      keep_prob:keep_prob, is_training:True})
            # train测试(这里仅测试训练集的一个batch)
            if i%train_log_step == 0:
                train_acc = sess.run(accuracy, feed_dict={input_images:batch_input_images,
                                                          input_labels: batch_input_labels,
#                                                          keep_prob:keep_prob, is_training: False})
                                                          is_training: False})
                print ("%s: Step [%d]  train Loss : %f, training accuracy :  %g" % (datetime.now(), i, train_loss, train_acc))

            # val测试(测试全部val数据)
            if i%val_log_step == 0:
                mean_loss, mean_acc=net_evaluation(sess, loss, accuracy, val_images_batch, val_labels_batch,val_nums)
                print ("%s: Step [%d]  val Loss : %f, val accuracy :  %g" % (datetime.now(), i, mean_loss, mean_acc))

            # 模型保存:每迭代snapshot次或者最后一次保存模型
            if (i %snapshot == 0 and i >0)or i == max_steps:
                print('-----save:{}-{}'.format(snapshot_prefix,i))
                saver.save(sess, snapshot_prefix, global_step=i)
            # 保存val准确率最高的模型
            if mean_acc>max_acc and mean_acc>0.8:
                max_acc=mean_acc
                path = os.path.dirname(snapshot_prefix)
                best_models=os.path.join(path,'best_models_{}_{:.4f}.ckpt'.format(i,max_acc))
                print('------save:{}'.format(best_models))
                saver.save(sess, best_models)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    train_record_file=['cifar10_extensions/cifar10_train_aug64.tfrecords_seg0',
                       'cifar10_extensions/cifar10_train_aug64.tfrecords_seg1',
                       'cifar10_extensions/cifar10_train_aug64.tfrecords_seg2',
                       'cifar10_extensions/cifar10_train_aug64.tfrecords_seg3',
                       'cifar10_extensions/cifar10_train_aug64.tfrecords_seg4',
                       'cifar10_extensions/cifar10_train_aug64.tfrecords_seg5',
                       'cifar10_extensions/cifar10_train_aug64.tfrecords_seg6',
                       'cifar10_extensions/cifar10_train_aug64.tfrecords_seg7',
                       'cifar10_extensions/cifar10_train_aug64.tfrecords_seg8',
                       'cifar10_extensions/cifar10_train_aug64.tfrecords_seg9']
#    train_record_file=['cifar10_extensions/cifar10_train.tfrecords_seg']
    val_record_file=['cifar10_extensions/cifar10_test.tfrecords_seg']

    train_log_step=200
    base_lr = 0.1  # 学习率
    max_steps = 2540000  # 迭代次数
    train_param=[base_lr,max_steps]

    val_log_step=500
    snapshot=20000#保存文件间隔
    snapshot_prefix='cifar10_extensions/vgg16/cifar_vgg16_aug160'
    train(train_record_file=train_record_file,
          train_log_step=train_log_step,
          train_param=train_param,
          val_record_file=val_record_file,
          val_log_step=val_log_step,
          labels_nums=labels_nums,
          data_shape=data_shape,
          snapshot=snapshot,
          snapshot_prefix=snapshot_prefix)
