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
import os
from datetime import datetime
import slim.nets.inception_v3 as inception_v3

from create_tf_record import get_example_nums,read_records,get_batch_images
import tensorflow.contrib.slim as slim

print("Tensorflow version:{}".format(tf.__version__))
labels_nums = 10  #  the number of labels
batch_size = 64  
epoch = 100
resize_height = 299  # imagenet size
resize_width = 299
depths = 3
data_shape = [batch_size, resize_height, resize_width, depths]


input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')
input_labels = tf.placeholder(dtype=tf.int32, shape=[None, labels_nums], name='label')
#for tensorflow1.2, keep_prob cannot be defined as placeholder, it must be a scalar and it needn't be fed to session
keep_prob = 0.5#tf.placeholder(tf.float32,name='keep_prob')
is_training = tf.placeholder(tf.bool, name='is_training')

def net_evaluation(sess,loss,accuracy,val_images_batch,val_labels_batch,val_nums):
    val_max_steps = int(val_nums / batch_size)
    val_losses = []
    val_accs = []
    for _ in range(val_max_steps):
        val_x, val_y = sess.run([val_images_batch, val_labels_batch])

        val_loss,val_acc = sess.run([loss,accuracy], feed_dict={input_images: val_x, input_labels: val_y, is_training: False})
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    mean_loss = np.array(val_losses, dtype=np.float32).mean()
    mean_acc = np.array(val_accs, dtype=np.float32).mean()
    return mean_loss, mean_acc

#def step_train(train_op,loss,accuracy,
def step_train(train_op,learning_rate,max_steps,loss,accuracy,
               train_images_batch,train_labels_batch,train_nums,train_log_step,
               val_images_batch,val_labels_batch,val_nums,val_log_step,
               snapshot_prefix,snapshot):
    '''
    循环迭代训练过程
    :param train_op: 训练op
    :param loss:     loss函数
    :param accuracy: 准确率函数
    :param train_images_batch: 训练images数据
    :param train_labels_batch: 训练labels数据
    :param train_nums:         总训练数据
    :param train_log_step:   训练log显示间隔
    :param val_images_batch: 验证images数据
    :param val_labels_batch: 验证labels数据
    :param val_nums:         总验证数据
    :param val_log_step:     验证log显示间隔
    :param snapshot_prefix: 模型保存的路径
    :param snapshot:        模型保存间隔
    :return: None
    '''
    saver = tf.train.Saver(max_to_keep=100)
#    saver = tf.compat.v1.train.Saver(max_to_keep=100)
    max_acc = 0.0
    
#    variables_to_restore=slim.get_variables_to_restore()
#    restorer=tf.train.Saver(variables_to_restore)    

    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
#        restorer.restore(sess,'inception_v3.ckpt')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(max_steps + 1):
            batch_input_images, batch_input_labels = sess.run([train_images_batch, train_labels_batch])
            _, lr, train_loss = sess.run([train_op, learning_rate, loss], feed_dict={input_images: batch_input_images,
                                                                  input_labels: batch_input_labels,
#                                                                  keep_prob: 0.5, is_training: True})
#for tensorflow1.2, keep_prob cannot be defined as placeholder, it must be a scalar and it needn't be fed to session
                                                                  is_training: True})
            # train测试(这里仅测试训练集的一个batch)
            if i % train_log_step == 0:
                train_acc = sess.run(accuracy, feed_dict={input_images: batch_input_images,
                                                          input_labels: batch_input_labels,
#                                                          keep_prob: 1.0, is_training: False})
#for tensorflow1.2, keep_prob cannot be defined as placeholder, it must be a scalar and it needn't be fed to session
                                                           is_training: False})
#                print("%s: Step [%d]  train Loss : %f, training accuracy :  %g, host_global_step: %d, lr: %f" % (
#                datetime.now(), i, train_loss, train_acc, host_global_step, lr))
                print("{}: Step {}  train Loss : {}, training accuracy :  {}, lr: {}".format (
                datetime.now(), i, train_loss, train_acc, lr))

            # val测试(测试全部val数据)
            if i % val_log_step == 0:
                mean_loss, mean_acc = net_evaluation(sess, loss, accuracy, val_images_batch, val_labels_batch, val_nums)
                print("%s: Step [%d]  val Loss : %f, val accuracy :  %g" % (datetime.now(), i, mean_loss, mean_acc))

            # 模型保存:每迭代snapshot次或者最后一次保存模型
            if (i % snapshot == 0 and i > 0) or i == max_steps:
                print('-----save:{}-{}'.format(snapshot_prefix, i))
                saver.save(sess, snapshot_prefix, global_step=i)
            # 保存val准确率最高的模型
            if mean_acc > max_acc and mean_acc > 0.7:
                max_acc = mean_acc
                path = os.path.dirname(snapshot_prefix)
                best_models = os.path.join(path, 'best_models_{}_{:.4f}.ckpt'.format(i, max_acc))
                print('------save:{}'.format(best_models))
                saver.save(sess, best_models)

        coord.request_stop()
        coord.join(threads)

def train(train_record_file,
          train_log_step,
#          train_param,
          val_record_file,
          val_log_step,
          labels_nums,
          data_shape,
#          snapshot,
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
#    [base_lr,max_steps]=train_param
    [batch_size,resize_height,resize_width,depths]=data_shape

    # 获得训练和测试的样本数
    if len(train_record_file)>1:
        #train_nums=0
        #for i in range(len(train_record_file)):
        #    train_nums_i=get_example_nums(train_record_file[i])
        #    train_nums=train_nums+train_nums_i
        train_nums=845000    
    else:
        train_nums=get_example_nums(train_record_file[0])
    val_nums=get_example_nums(val_record_file)
    print('train nums:%d,val nums:%d'%(train_nums,val_nums))

    # 从record中读取图片和labels数据
    # train数据,训练数据一般要求打乱顺序shuffle=True
#    train_images, train_labels = read_records(train_record_file, resize_height, resize_width, type='normalization')
    train_images, train_labels = read_records(train_record_file, resize_height, resize_width, type='centralization',padding=True,crop=True,flip=True)
    train_images_batch, train_labels_batch = get_batch_images(train_images, train_labels,
                                                              batch_size=batch_size, labels_nums=labels_nums,
                                                              one_hot=True, shuffle=True)
    # test data, don't need to be shuffled
#    val_images, val_labels = read_records([val_record_file], resize_height, resize_width, type='normalization')
    val_images, val_labels = read_records([val_record_file], resize_height, resize_width, type='centralization')
    val_images_batch, val_labels_batch = get_batch_images(val_images, val_labels,
                                                          batch_size=batch_size, labels_nums=labels_nums,
                                                          one_hot=True, shuffle=False)

    # Define the model:
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        out, end_points = inception_v3.inception_v3(inputs=input_images, num_classes=labels_nums, dropout_keep_prob=keep_prob, is_training=is_training)


    tf.losses.softmax_cross_entropy(onehot_labels=input_labels, logits=out)
    loss = tf.losses.get_total_loss(add_regularization_losses=True)
    
    global_step = tf.Variable(0, trainable=False)
#    learning_rate = tf.train.exponential_decay(0.05, global_step, 150, 0.9)
    max_steps = train_nums/batch_size*epoch
    max_steps=int(max_steps)
    snapshot = train_nums/batch_size*2
    learning_rate = tf.train.exponential_decay(0.045, global_step, int(train_nums/batch_size*2), 0.94, staircase=True)
    add_global = global_step.assign_add(1)
    #
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    # # train_tensor = optimizer.minimize(loss, global_step)
    # train_op = slim.learning.create_train_op(loss, optimizer,global_step=global_step)
    
    # 在定义训练的时候, 注意到我们使用了`batch_norm`层时,需要更新每一层的`average`和`variance`参数,
    # 更新的过程不包含在正常的训练过程中, 需要我们去手动像下面这样更新
    # 通过`tf.get_collection`获得所有需要更新的`op`
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # 使用`tensorflow`的控制流, 先执行更新算子, 再执行训练
    with tf.control_dependencies(update_ops):
        # create_train_op that ensures that when we evaluate it to get the loss,
        # the update_ops are done and the gradient updates are computed.
        # train_op = slim.learning.create_train_op(total_loss=loss,optimizer=optimizer)
  #      train_op = slim.learning.create_train_op(total_loss=loss, optimizer=optimizer)
#        train_op = slim.learning.create_train_op(total_loss=loss, optimizer=optimizer, global_step=global_step)
        train_op = slim.learning.create_train_op(total_loss=loss, optimizer=optimizer, global_step=global_step, clip_gradient_norm=2.)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(input_labels, 1)), tf.float32))
    # 循环迭代过程
#    step_train(train_op, loss, accuracy,
    step_train(train_op, learning_rate, max_steps, loss, accuracy,
               train_images_batch, train_labels_batch, train_nums, train_log_step,
               val_images_batch, val_labels_batch, val_nums, val_log_step,
               snapshot_prefix, snapshot)


if __name__ == '__main__':
#    train_record_file=['dataset/record/train299_first10cls.tfrecords']

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
    val_record_file='data/caffe_ilsvrc12_record/val299.tfrecords'

    train_log_step=100
#    base_lr = 0.01  # learning rate

    val_log_step=200
#    snapshot=2000#保存文件间隔
    snapshot_prefix='models/incepv3_consdef_fisrt10cls_100epos.ckpt'
    train(train_record_file=train_record_file,
          train_log_step=train_log_step,
#          train_param=train_param,
          val_record_file=val_record_file,
          val_log_step=val_log_step,
          labels_nums=labels_nums,
          data_shape=data_shape,
#          snapshot=snapshot,
          snapshot_prefix=snapshot_prefix)
