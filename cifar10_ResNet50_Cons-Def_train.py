"""
This file used to train a model using augmented images with TensorFlow.
The augmentations are pre-processed after run cifar10_creat_tf_record.py
Revised from the work of PanJinquan: https://github.com/PanJinquan/tensorflow_models_learning
Xintao Ding
School of Computer and Information, Anhui Normal University
xintaoding@163.com
"""

import tensorflow as tf 
import numpy as np 
import os
from datetime import datetime
import slim.nets.resnet_v1 as resnet_v1
from cifar10_create_tf_record import get_example_nums, read_records, get_batch_images
import tensorflow.contrib.slim as slim

labels_nums = 10  #  the number of labels
batch_size = 80  #
resize_height = 32 # Cifar10 size
resize_width = 32   
net_height= 128#resnet size
net_width = 128
depths = 3
data_shape = [batch_size, resize_height, resize_width, depths]

input_images = tf.placeholder(dtype=tf.float32, shape=[None, net_height, net_width, depths], name='input')
input_labels = tf.placeholder(dtype=tf.int32, shape=[None, labels_nums], name='label')

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

def step_train(train_op,loss,accuracy,
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
    max_acc = 0.8782
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        ckpt=tf.train.get_checkpoint_state('cifar10_extensions/resnet50_64x_aug_train2400k_iters')
        if ckpt and ckpt.model_checkpoint_path:
           saver.restore(sess,ckpt.model_checkpoint_path)


        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(max_steps + 1):
            batch_input_images, batch_input_labels = sess.run([train_images_batch, train_labels_batch])
            _, train_loss = sess.run([train_op, loss], feed_dict={input_images: batch_input_images,
                                                                  input_labels: batch_input_labels,
#                                                                  keep_prob: 0.8, is_training: True})
                                                                  is_training: True})
            # train测试(这里仅测试训练集的一个batch)
            if i % train_log_step == 0:
                train_acc = sess.run(accuracy, feed_dict={input_images: batch_input_images,
                                                          input_labels: batch_input_labels,
#                                                          keep_prob: 1.0, is_training: False})
                                                          is_training: False})
                print("%s: Step [%d]  train Loss : %f, training accuracy :  %g" % (
                datetime.now(), i, train_loss, train_acc))

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
                                              type='normalization', crop_flip=True)#cifar10 is only required to be ranged in [0,1] for normalization
#                                              type='centralization', crop_flip=True)#for imagenet, input should be centerlized
    train_images_batch, train_labels_batch = get_batch_images(train_images, train_labels,
                                                              batch_size=batch_size, labels_nums=labels_nums,
                                                              one_hot=True, shuffle=True)
    train_images_batch = tf.image.resize_images(train_images_batch,size=(net_height, net_width))

    # val数据,验证数据可以不需要打乱数据
    val_images, val_labels = read_records(val_record_file, resize_height, resize_width, depths,
#                                              net_height=net_height, net_width = net_width,
                                              type='normalization')
#                                              type='centralization', crop_flip=True)
    val_images_batch, val_labels_batch = get_batch_images(val_images, val_labels,
                                                          batch_size=batch_size, labels_nums=labels_nums,
                                                          one_hot=True, shuffle=False)
    val_images_batch = tf.image.resize_images(val_images_batch,size=(net_height, net_width))

    # Define the model:
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    #    out, end_points = resnet_v1.resnet_v1_101(inputs=input_images, num_classes=labels_nums, is_training=is_training,global_pool=True)
        out, end_points = resnet_v1.resnet_v1_50(inputs=input_images, num_classes=labels_nums, is_training=is_training,global_pool=True)

    tf.losses.softmax_cross_entropy(onehot_labels=input_labels, logits=out)  
    loss = tf.losses.get_total_loss(add_regularization_losses=True)  

    # Specify the optimization scheme:

    # 在定义训练的时候, 注意到我们使用了`batch_norm`层时,需要更新每一层的`average`和`variance`参数,
    # 更新的过程不包含在正常的训练过程中, 需要我们去手动像下面这样更新
    # 通过`tf.get_collection`获得所有需要更新的`op`
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # 使用`tensorflow`的控制流, 先执行更新算子, 再执行训练
    with tf.control_dependencies(update_ops):
#        print("update_ops:{}".format(update_ops))
        # create_train_op that ensures that when we evaluate it to get the loss,
        # the update_ops are done and the gradient updates are computed.
        # train_op = tf.train.MomentumOptimizer(learning_rate=base_lr, momentum=0.9).minimize(loss)
        train_op = tf.train.AdadeltaOptimizer(learning_rate=base_lr).minimize(loss)


    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(input_labels, 1)), tf.float32))
    # 循环迭代过程
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
    base_lr = 0.1#0.001  # 学习率
    max_steps = 2540000  # 迭代次数50000*65*100/batch_size
#    max_steps = 1000000  # 迭代次数50000*100/batch_size
    train_param=[base_lr,max_steps]

    val_log_step=500
    snapshot=20000#保存文件间隔
    snapshot_prefix='models/cifar10/resnet50_consdef'
    train(train_record_file=train_record_file,
          train_log_step=train_log_step,
          train_param=train_param,
          val_record_file=val_record_file,
          val_log_step=val_log_step,
          labels_nums=labels_nums,
          data_shape=data_shape,
          snapshot=snapshot,
          snapshot_prefix=snapshot_prefix)
