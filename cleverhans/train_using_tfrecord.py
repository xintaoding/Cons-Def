"""
Multi-replica synchronous training


NOTE: This module is much more free to change than many other modules
in CleverHans. CleverHans is very conservative about changes to any
code that affects the output of benchmark tests (attacks, evaluation
methods, etc.). This module provides *model training* functionality
not *benchmarks* and thus is free to change rapidly to provide better
speed, accuracy, etc.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import time
import warnings

import math
import numpy as np
from six.moves import xrange
import tensorflow as tf

from cleverhans import canary
from cleverhans.utils import _ArgsWrapper, create_logger
from cleverhans.utils import safe_zip
from cleverhans.utils_tf import infer_devices
from cleverhans.utils_tf import initialize_uninitialized_global_variables

from cifar10_extensions.read_and_decode import read_and_decode

from tensorflow.python.keras.utils import np_utils




_logger = create_logger("train")
_logger.setLevel(logging.INFO)


def train(sess, loss, x_train=None, y_train=None,
          init_all=False, evaluate=None, feed=None, args=None,
          rng=None, var_list=None, fprop_args=None, optimizer=None,
          devices=None, x_batch_preprocessor=None, use_ema=False,
          ema_decay=.998, run_canary=None,
          loss_threshold=1e5, dataset_train=None, dataset_size=None):
  """
  Run (optionally multi-replica, synchronous) training to minimize `loss`
  :param sess: TF session to use when training the graph
  :param loss: tensor, the loss to minimize
  :param x_train: numpy array with training inputs or tf Dataset
  :param y_train: numpy array with training outputs or tf Dataset
  :param init_all: (boolean) If set to true, all TF variables in the session
                   are (re)initialized, otherwise only previously
                   uninitialized variables are initialized before training.
  :param evaluate: function that is run after each training iteration
                   (typically to display the test/validation accuracy).
  :param feed: An optional dictionary that is appended to the feeding
               dictionary before the session runs. Can be used to feed
               the learning phase of a Keras model for instance.
  :param args: dict or argparse `Namespace` object.
               Should contain `nb_epochs`, `learning_rate`,
               `batch_size`
  :param rng: Instance of numpy.random.RandomState
  :param var_list: Optional list of parameters to train.
  :param fprop_args: dict, extra arguments to pass to fprop (loss and model).
  :param optimizer: Optimizer to be used for training
  :param devices: list of device names to use for training
      If None, defaults to: all GPUs, if GPUs are available
                            all devices, if no GPUs are available
  :param x_batch_preprocessor: callable
      Takes a single tensor containing an x_train batch as input
      Returns a single tensor containing an x_train batch as output
      Called to preprocess the data before passing the data to the Loss
  :param use_ema: bool
      If true, uses an exponential moving average of the model parameters
  :param ema_decay: float or callable
      The decay parameter for EMA, if EMA is used
      If a callable rather than a float, this is a callable that takes
      the epoch and batch as arguments and returns the ema_decay for
      the current batch.
  :param loss_threshold: float
      Raise an exception if the loss exceeds this value.
      This is intended to rapidly detect numerical problems.
      Sometimes the loss may legitimately be higher than this value. In
      such cases, raise the value. If needed it can be np.inf.
  :param dataset_train: tf Dataset instance.
      Used as a replacement for x_train, y_train for faster performance.
    :param dataset_size: integer, the size of the dataset_train.
  :return: True if model trained
  """

  # Check whether the hardware is working correctly
  canary.run_canary()
  if run_canary is not None:
    warnings.warn("The `run_canary` argument is deprecated. The canary "
                  "is now much cheaper and thus runs all the time. The "
                  "canary now uses its own loss function so it is not "
                  "necessary to turn off the canary when training with "
                  " a stochastic loss. Simply quit passing `run_canary`."
                  "Passing `run_canary` may become an error on or after "
                  "2019-10-16.")

  args = _ArgsWrapper(args or {})
  fprop_args = fprop_args or {}

  # Check that necessary arguments were given (see doc above)
  # Be sure to support 0 epochs for debugging purposes
  if args.nb_epochs is None:
    raise ValueError("`args` must specify number of epochs")
  if optimizer is None:
    if args.learning_rate is None:
      raise ValueError("Learning rate was not given in args dict")
  assert args.batch_size, "Batch size was not given in args dict"

  if rng is None:
    rng = np.random.RandomState()

  if optimizer is None:
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
  else:
    if not isinstance(optimizer, tf.train.Optimizer):
      raise ValueError("optimizer object must be from a child class of "
                       "tf.train.Optimizer")

  grads = []
  xs = []
  preprocessed_xs = []
  ys = []
  if dataset_train is not None:
    assert x_train is None and y_train is None and x_batch_preprocessor is None
    if dataset_size is None:
      raise ValueError("You must provide a dataset size")
  
#  assert 1==2   
  #https://www.cnblogs.com/ywheunji/p/11390219.html

  devices = infer_devices(devices)
  nb_classes=10
  for device in devices:
    with tf.device(device):
      x = tf.placeholder(dtype=tf.float32, shape=(None,32, 32, 3))
      y = tf.placeholder(dtype=tf.float32, shape=(None,nb_classes))
      xs.append(x)
      ys.append(y)

      if x_batch_preprocessor is not None:
        x = x_batch_preprocessor(x)

      # We need to keep track of these so that the canary can feed
      # preprocessed values. If the canary had to feed raw values,
      # stochastic preprocessing could make the canary fail.
      preprocessed_xs.append(x)

      loss_value = loss.fprop(x, y, **fprop_args)

      grads.append(optimizer.compute_gradients(
          loss_value, var_list=var_list))
  num_devices = len(devices)
  print("num_devices: ", num_devices)

  grad = avg_grads(grads)
  print("use_ema:{},grads:{},grad:{}".format(use_ema,type(grads),type(grad)))
#  assert 1==2
  # Trigger update operations within the default graph (such as batch_norm).
  with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_step = optimizer.apply_gradients(grad)



  batch_size = args.batch_size

  assert batch_size % num_devices == 0


  if init_all:
    sess.run(tf.global_variables_initializer())
  else:
    initialize_uninitialized_global_variables(sess)
  print("init_all===========:{},dataset_train:{}".format(init_all,dataset_train))
#  assert 1==2
  saver = tf.train.Saver()
  filename_queue = ['./cifar10_extensions/cifar10_train_aug64.tfrecords_seg0',
        './cifar10_extensions/cifar10_train_aug64.tfrecords_seg1',
        './cifar10_extensions/cifar10_train_aug64.tfrecords_seg2',
        './cifar10_extensions/cifar10_train_aug64.tfrecords_seg3',
        './cifar10_extensions/cifar10_train_aug64.tfrecords_seg4',
        './cifar10_extensions/cifar10_train_aug64.tfrecords_seg5',
        './cifar10_extensions/cifar10_train_aug64.tfrecords_seg6',
        './cifar10_extensions/cifar10_train_aug64.tfrecords_seg7',
        './cifar10_extensions/cifar10_train_aug64.tfrecords_seg8',
        './cifar10_extensions/cifar10_train_aug64.tfrecords_seg9']
#  for epoch in xrange(args.nb_epochs):
  img,label=read_and_decode(filename_queue, args.nb_epochs, batch_size, 32, 32, 3)
  sess.run(tf.local_variables_initializer())
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess,coord=coord)
  iter_count=0
  jihao = 1
  epoch0=0
  try:
      while not coord.should_stop():
          iter_count=iter_count+1
          if jihao:
              jihao = 0
              prev = time.time()
          img_host,label_host=sess.run([img,label])
          label_host = np_utils.to_categorical(label_host, nb_classes)
          feed_dict = {x: img_host, y: label_host}
          _, loss_numpy = sess.run([train_step, loss_value],feed_dict=feed_dict)
#          if iter_count%500==0:
#              _logger.info("iter:"+str(iter_count)+ " iterations ")
          epoch = (iter_count*batch_size)//(dataset_size*65)
          if epoch>epoch0:
              epoch0 = epoch
              jihao = 1
              cur = time.time()
              if evaluate is not None:
                  evaluate()
              _logger.info("epoch:"+str(epoch0)+ " took " + str(cur - prev) + " seconds")
              if  epoch==39 or epoch==44 or epoch==49 or epoch==54 or epoch==59 or epoch==64 or epoch==69 or epoch==74 or epoch==79 or epoch==84 or epoch==89 or epoch==94 or epoch==99:
                  saver.save(sess,'./cifar10_clean_train_epoch'+str(epoch))

  except   tf.errors.OutOfRangeError:
        _logger.info('An epoch training is done')
  finally:
        coord.request_stop()   
  coord.join(threads)
        
  
  

  return True


def avg_grads(tower_grads):
  """Calculate the average gradient for each shared variable across all
  towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been
     averaged across all towers.

  Modified from this tutorial: https://tinyurl.com/n3jr2vm
  """
  if len(tower_grads) == 1:
    return tower_grads[0]
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = [g for g, _ in grad_and_vars]

    # Average over the 'tower' dimension.
    grad = tf.add_n(grads) / len(grads)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    assert all(v is grad_and_var[1] for grad_and_var in grad_and_vars)
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads
