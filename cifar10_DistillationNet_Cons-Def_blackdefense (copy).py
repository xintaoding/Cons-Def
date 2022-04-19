"""
This tutorial shows how to generate transferable attacks from adversarial examples of the source model
The target model is CNN that is structured in 4 convolutional layers, 2 pooling layers, and 3 fully connected layers.
The target model requires the input with the size of 32*32, therefore, the source examples are resized for implementation.

Xintao Ding
School of Computer and Information, Anhui Normal University
xintaoding@163.com
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf

from cleverhans.attacks import FastGradientMethod
from cleverhans.augmentation import random_horizontal_flip, random_shift
from cleverhans.compat import flags
from cleverhans.dataset import CIFAR10
from cleverhans.loss import CrossEntropy
#from cleverhans.model_zoo.all_convolutional import ModelAllConvolutional
from cleverhans.model_zoo.four_conv2FC import Model4Convolutional2FC#Revised by Ding
from cleverhans.utils_tf import model_eval
from cleverhans.data_extenv2 import data_exten#Added by Ding

#from cleverhans.data_exten_mulpro import data_exten
from sklearn.metrics import roc_curve, roc_auc_score#Added by Ding

import matplotlib.pyplot as plt
from PIL import Image
import cv2

import multiprocessing

FLAGS = flags.FLAGS

BATCH_SIZE = 128
LEARNING_RATE = 0.001
BACKPROP_THROUGH_ATTACK = False
NB_FILTERS = 64

def cifar10_tutorial(train_start=0, train_end=50000, test_start=0,
                     test_end=10000, batch_size=BATCH_SIZE,
                     learning_rate=LEARNING_RATE,
                     testing=False,
                     backprop_through_attack=BACKPROP_THROUGH_ATTACK,
                     nb_filters=NB_FILTERS, num_threads=None,
                     label_smoothing=0.1):
  """
  CIFAR10 cleverhans tutorial
  :param train_start: index of first training set example
  :param train_end: index of last training set example
  :param test_start: index of first test set example
  :param test_end: index of last test set example
  :param nb_epochs: number of epochs to train model
  :param batch_size: size of training batches
  :param learning_rate: learning rate for training
  :param clean_train: perform normal training on clean examples only
                      before performing adversarial training.
  :param testing: if true, complete an AccuracyReport for unit tests
                  to verify that performance is adequate
  :param backprop_through_attack: If True, backprop through adversarial
                                  example construction process during
                                  adversarial training.
  :param label_smoothing: float, amount of label smoothing for cross entropy
  :return: an AccuracyReport object
  """

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  # Create TF session
  if num_threads:
    config_args = dict(intra_op_parallelism_threads=1)
  else:
    config_args = {}
  sess = tf.Session(config=tf.ConfigProto(**config_args))

  # Get CIFAR10 data
  data = CIFAR10(train_start=train_start, train_end=train_end,
                 test_start=test_start, test_end=test_end)
  dataset_size = data.x_train.shape[0]
  dataset_train = data.to_tensorflow()[0]
  dataset_train = dataset_train.map(
      lambda x, y: (random_shift(random_horizontal_flip(x)), y), 4)
  dataset_train = dataset_train.batch(batch_size)
  dataset_train = dataset_train.prefetch(16)
  x_train, y_train = data.get_set('train')
  x_test, y_test = data.get_set('test')
  
#  plt.imshow(np.uint8(x_test[0,:,:,:]*255))

  # Use Image Parameters
  img_rows, img_cols, nchannels = x_test.shape[1:4]
  nb_classes = y_test.shape[1]

  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))

  eval_params = {'batch_size': batch_size}

  net_height=32
  net_width=32
#    model = ModelAllConvolutional('model1', nb_classes, nb_filters,
#                                  input_shape=[32, 32, 3])
  model = Model4Convolutional2FC('model1', nb_classes, nb_filters, input_shape=[net_height, net_width, nchannels])
  preds = model.get_logits(x)
  probs = model.get_probs(x)

#########################################Added by Ding  
  saver = tf.train.Saver()
  saver.restore(sess,'./models/cifar10/cifar10_dtcnn_train_epoch89')#target model
  print("aaaaaaaaaaaaaaaaa=+++++++++++++++++++++++++++++")
#########################################


#  np.save("cifar10_DistillationNet_augmodel_fgsmlinf_10000adv",adv)#save advs produced on clean model:
  adv = np.load("cifar10_vgg16_augmodel_pgd_10000adv.npy")#source examples for attack

  x_tt =x_train[:100,:,:,:]
  y_tt = y_train[:100,:]
  dat0ext=data_exten(x_tt, y_tt, 100, base_range=4)
  
  print("advaaaaaaaaaaaaaaaaaaaaaaaaaaa:{},{}".format(np.sum(adv-x_test),adv.shape))
  
# Evaluate the accuracy of the model on benign and adversarial examples
  accuracy,suc_att_exam = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
  print('Test accuracy on legitimate test examples: {}'.format(accuracy))
  adv_accuracy,adv_suc_att_exam = model_eval(sess, x, y, preds, adv, y_test, args=eval_params)
  print('Test accuracy on adversarial test examples: {}'.format(adv_accuracy))
  print('Test Attack Successful Rate (ASR) on examples: {0:.4f}' .format (1-adv_accuracy))
  
 #for untargeted attack, suc_att_exam[i] is true means a successful classified examples
 #for targeted attack, suc_att_exam[i] is true means a successful attack, it counts succeful attacked examples
  percent_perturbed = np.mean(np.sum((adv - x_test)**2, axis=(1, 2, 3))**.5)

  dsae=0
  kk=0
  for i in range(len(adv_suc_att_exam)):
      if adv_suc_att_exam[i]==0 and suc_att_exam[i]>0:#adversarial is misclassified but its corresponding binign example is correctly detected
          dsae+=np.sum((adv[i,:,:,:] - x_test[i,:,:,:])**2)**.5
          kk += 1
  dsae=dsae/kk
  print("For untargeted attack, the number of misclassified examples (successful attack), sum(adv_suc_att_exam==0):{}, dsae:{}".format(sum(adv_suc_att_exam==0),dsae))

  print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))
  print('The number of successful attack:{}, Avg. L_2 norm of perturbations on successful attack / dsae:{}'.format(kk,dsae))
 
      
  pad_size=4
  x_test=np.pad(x_test,((0,0),(pad_size,pad_size),(pad_size,pad_size),(0,0)),'reflect')
  x_testcrop = np.zeros((len(x_test),net_height,net_width,3),dtype=np.float32)
  adv = np.round(adv*256)/256.0
  adv = np.pad(adv,((0,0),(pad_size,pad_size),(pad_size,pad_size),(0,0)),'reflect')
  advcrop = np.zeros((len(adv),net_height,net_width,3),dtype=np.float32)
  for i in range(len(adv)):
      tf_image = adv[i,:,:,:]
      test_image = x_test[i,:,:,:]
      lu1 = np.random.randint(0,pad_size*2)
      lu2 = np.random.randint(0,pad_size*2)
      advcrop[i,:,:,:] = tf_image[lu1:lu1+net_height,lu2:lu2+net_width,:]
      x_testcrop[i,:,:,:] = test_image[lu1:lu1+net_height,lu2:lu2+net_width,:]
  adv = advcrop
  x_test = x_testcrop
   
  batch_size = 500  #
  base_range=4
  n_pert = base_range**nchannels
  ext_bat = n_pert+1
  
  logits_ext = np.zeros((test_end*n_pert,nb_classes),dtype=np.float32)
  logits_adv_ext = np.zeros((test_end*n_pert,nb_classes),dtype=np.float32)
  test_prob_pertpart=np.zeros((test_end*n_pert,nb_classes),dtype=np.float32)
  adv_prob_pertpart=np.zeros((test_end*n_pert,nb_classes),dtype=np.float32)
  y_test_pertpart = np.zeros((test_end*n_pert,nb_classes),dtype=np.float32)
  y_adv_pertpart = np.zeros((test_end*n_pert,nb_classes),dtype=np.float32)
  x_adv_pertpart = np.zeros((batch_size*n_pert*2,net_height,net_width,nchannels),dtype=np.float32)
  x_test_pertpart = np.zeros((batch_size*n_pert*2,net_height,net_width,nchannels),dtype=np.float32)
  val_max_steps = int(len(adv) / batch_size/2)
    
  adv_prob_legit = np.zeros((test_end,nb_classes),dtype=np.float32)
  test_prob_legit = np.zeros((test_end,nb_classes),dtype=np.float32)
    

  # Close TF session
  sess.close()



def main(argv=None):
  from cleverhans_tutorials import check_installation
  check_installation(__file__)

  cifar10_tutorial(batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters)


if __name__ == '__main__':
  flags.DEFINE_integer('nb_filters', NB_FILTERS,
                       'Model size multiplier')
  flags.DEFINE_integer('batch_size', BATCH_SIZE,
                       'Size of training batches')
  flags.DEFINE_float('learning_rate', LEARNING_RATE,
                     'Learning rate for training')
  flags.DEFINE_bool('backprop_through_attack', BACKPROP_THROUGH_ATTACK,
                    ('If True, backprop through adversarial example '
                     'construction process during adversarial training'))

  tf.app.run()
