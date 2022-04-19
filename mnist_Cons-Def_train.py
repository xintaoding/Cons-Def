"""
This file used to train a model using augmented images with TensorFlow.
The augmentations are serially processed in data_exten.py
Revised from Cleverhans
Xintao Ding
School of Computer and Information, Anhui Normal University
xintaoding@163.com
"""#coding=utf-8

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import tensorflow as tf

from cleverhans.compat import flags
from cleverhans.loss import CrossEntropy
from cleverhans.dataset import MNIST
from cleverhans.utils_tf import model_eval
from cleverhans.train import train
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.model_zoo.basic_cnn import ModelBasicCNN
#from cleverhans.model_zoo.four_conv2FC import Model4Convolutional2FC#Revised by Ding
from cleverhans.data_exten import data_exten#intensity exchange-based data augmentation
from sklearn.metrics import roc_curve, roc_auc_score#Added by Ding

FLAGS = flags.FLAGS

NB_EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.001
BACKPROP_THROUGH_ATTACK = False
NB_FILTERS = 64

def mnist_tutorial(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                   learning_rate=LEARNING_RATE,
#                   testing=True,
                   backprop_through_attack=BACKPROP_THROUGH_ATTACK,
                   nb_filters=NB_FILTERS, num_threads=None,
                   label_smoothing=0.1):
  """
  MNIST cleverhans tutorial
  :param train_start: index of first training set example
  :param train_end: index of last training set example
  :param test_start: index of first test set example
  :param test_end: index of last test set example
  :param nb_epochs: number of epochs to train model
  :param batch_size: size of training batches
  :param learning_rate: learning rate for training
  :param backprop_through_attack: If True, backprop through adversarial
                                  example construction process during
                                  adversarial training.
  :param label_smoothing: float, amount of label smoothing for cross entropy
  :return: an AccuracyReport object
  """

  # Object used to keep track of (and return) key accuracies
  report = AccuracyReport()

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)
  np.random.seed(1234)

  # Set logging level to see debug information
  set_log_level(logging.DEBUG)

  # Create TF session
  if num_threads:
    config_args = dict(intra_op_parallelism_threads=1)
  else:
    config_args = {}
  sess = tf.Session(config=tf.ConfigProto(**config_args))

  # Get MNIST data
  mnist = MNIST(train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end)
  x_train, y_train = mnist.get_set('train')
  
  ###############extend train dataset
  print("x_train:{}".format(x_train.shape))
  
  x_test, y_test = mnist.get_set('test')
  
  # Use Image Parameters
  print("y_train_shape:{}".format(y_train.shape))#########################################
  img_rows, img_cols, nchannels = x_train.shape[1:4]
  nb_classes = y_train.shape[1]

# augment training set======Added by Ding
  x_train, y_train = data_exten(x_train, y_train, train_end, nb_classes, img_rows, img_cols, 1)

  print("nb_classes:{}".format(nb_classes))#########################################

  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                        nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))

  # Train an MNIST model
  train_params = {
      'nb_epochs': nb_epochs,
      'batch_size': batch_size,
      'learning_rate': learning_rate
  }
  eval_params = {'batch_size': batch_size}
  rng = np.random.RandomState([2017, 8, 30])

  def do_eval(x_set, y_set, preds, report_key, is_adv=None):
    acc, suc_att_exam = model_eval(sess, x, y, preds, x_set, y_set, args=eval_params)
    setattr(report, report_key, acc)
    if is_adv is None:
      report_text = None
    elif is_adv:
      report_text = 'adversarial'
    else:
      report_text = 'legitimate'
    if report_text:
      print('Test accuracy on %s examples: %0.4f' % (report_text, acc))
    return acc, suc_att_exam

  model = ModelBasicCNN('model1', nb_classes, nb_filters)
#    model = Model4Convolutional2FC('model1', nb_classes, nb_filters, input_shape=[28, 28, 1])
  preds = model.get_logits(x)
#    loss = CrossEntropy(model, smoothing=label_smoothing)
  loss = CrossEntropy(model)
  saver = tf.train.Saver()

  def evaluate():
      do_eval(x_test, y_test, preds, 'clean_train_clean_eval', False)
      
  #########################################Added by Ding  
  print("aaaaaaaaaaaaaaaaa=************************************")
  train(sess, loss, x_train, y_train, evaluate=evaluate,
          args=train_params, rng=rng, var_list=model.get_params())
  saver.save(sess,'./models/mnist_models/mnist_train_2_4_8_16_32Aug50iters')
  print("aaaaaaaaaaaaaaaaa=+++++++++++++++++++++++++++++")
  #########################################

# Evaluate the accuracy of the MNIST model on adversarial examples
  accuracy,suc_att_exam = do_eval(x_test, y_test, preds, 'clean_train_adv_eval', False)
    
def main(argv=None):
  """
  Run the tutorial using command line flags.
  """
  from cleverhans_tutorials import check_installation
  check_installation(__file__)

  mnist_tutorial(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                 learning_rate=FLAGS.learning_rate,
                 backprop_through_attack=FLAGS.backprop_through_attack,
                 nb_filters=FLAGS.nb_filters)


if __name__ == '__main__':
  flags.DEFINE_integer('nb_filters', NB_FILTERS,
                       'Model size multiplier')
  flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                       'Number of epochs to train model')
  flags.DEFINE_integer('batch_size', BATCH_SIZE,
                       'Size of training batches')
  flags.DEFINE_float('learning_rate', LEARNING_RATE,
                     'Learning rate for training')
  flags.DEFINE_bool('backprop_through_attack', BACKPROP_THROUGH_ATTACK,
                    ('If True, backprop through adversarial example '
                     'construction process during adversarial training'))

  tf.app.run()
