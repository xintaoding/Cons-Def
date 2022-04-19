"""
This tutorial shows how to implement Cons-Def against FGSM white-box attacks.
Xintao Ding
School of Computer and Information, Anhui Normal University
xintaoding@163.com
"""
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
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.model_zoo.basic_cnn import ModelBasicCNN
#from cleverhans.model_zoo.four_conv2FC import Model4Convolutional2FC#Revised by Ding
from cleverhans.data_exten import data_exten
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
  :param clean_train: perform normal training on clean examples only
                      before performing adversarial training.
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


  print("nb_classes:{}".format(nb_classes))#########################################

  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                        nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))

  # Train an MNIST model
  eval_params = {'batch_size': batch_size}
  fgsm_params = {
#      'ord': 1,
      'eps': 0.3,
      'clip_min': 0.,
      'clip_max': 1.
  }

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
  probs = model.get_probs(x)
#    loss = CrossEntropy(model, smoothing=label_smoothing)
  loss = CrossEntropy(model)
  saver = tf.train.Saver()

  def evaluate():
      do_eval(x_test, y_test, preds, 'clean_train_clean_eval', False)

  saver.restore(sess,'./models/mnist_models/mnist_train_2_4_8_16_32Aug50iters')
  print("aaaaaaaaaaaaaaaaa=************************************")
  #########################################

    # Initialize the Fast Gradient Sign Method (FGSM) attack object and
    # graph
  fgsm = FastGradientMethod(model, sess=sess)
  adv_x, grad_x, loss_x, _, _ = fgsm.generate(x, **fgsm_params)
  preds_adv = model.get_logits(adv_x)

  feed_dict = {x: x_test}
  adv = sess.run(adv_x,feed_dict=feed_dict)
  grad = sess.run(grad_x,feed_dict=feed_dict)
  loss = sess.run(loss_x,feed_dict=feed_dict)
  
#  np.save("mnist_fgsmadv_x_test",adv)#save fgsm adversarial examples produced on clean model
  print("advaaaaaaaaaaaaaaaaaaaaaaaaaaa:{},{},grad:{},loss:{}".format(np.sum(adv-x_test),adv.shape,np.sum(grad),loss))

    
# Evaluate the accuracy of the MNIST model on adversarial examples
  accuracy,suc_att_exam = do_eval(x_test, y_test, preds, 'clean_train_adv_eval', False)
  adv_accuracy,adv_suc_att_exam = do_eval(adv, y_test, preds, 'clean_train_adv_eval', True)
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
#  print("For targeted attack, the number of misclassified examples: sum(adv_suc_att_exam):{}, dsae:{}".format(sum(adv_suc_att_exam),dsae))
  print("For untargeted attack, the number of misclassified examples (successful attack), sum(adv_suc_att_exam==0):{}, dsae:{}".format(sum(adv_suc_att_exam==0),dsae))
  print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))
  print('The number of successful attack:{}, Avg. L_2 norm of perturbations on successful attack / dsae:{}'.format(kk,dsae))
###########################################################################    
#  adv = np.round(adv*256)/256.0
  adv34 = np.zeros((len(adv),6+img_rows,6+img_rows,1),dtype=np.float32)
  advcrop = np.zeros((len(adv),img_rows,img_cols,1),dtype=np.float32)
  adv34[:,3:31,3:31,:] = adv
  for i in range(len(adv)):
      tf_image = adv34[i,:,:,:]
      lu1 = np.random.randint(0,6)
      lu2 = np.random.randint(0,6)
      advcrop[i,:,:,:] = tf_image[lu1:lu1+img_rows,lu2:lu2+img_cols,:]
  adv = advcrop
###########################################################################    

  n_pert = 5#5 aug  test
  eva_thresh = np.linspace(3,5,3).astype('int32')#5 aug test
  x_adv_extended, y_adv_extended = data_exten(adv, y_test, test_end, nb_classes, img_rows, img_cols, 1)#
  x_adv_pertpart = x_adv_extended[:n_pert*test_end,:,:,:]
  y_adv_pertpart = y_adv_extended[:n_pert*test_end,:]
  x_test_extended, y_test_extended = data_exten(x_test, y_test, test_end, nb_classes, img_rows, img_cols, 1)
  x_test_pertpart = x_test_extended[:n_pert*test_end,:,:,:]
  y_test_pertpart = y_test_extended[:n_pert*test_end,:]

  print("Test accuracy on images extened by x_test")
  do_eval(x_test_pertpart, y_test_pertpart, preds, 'clean_train_adv_eval', False)
  print("fgsm attack on images extened by adv of x_test")
  do_eval(x_adv_pertpart, y_adv_pertpart, preds_adv, 'clean_train_adv_eval', True)

  feed_dict = {x: x_test_pertpart}
  y_pertpart_pred = sess.run(preds,feed_dict=feed_dict)
  y_pertpart_prob = sess.run(probs,feed_dict=feed_dict)
  auc_score_test = roc_auc_score(y_test_pertpart, y_pertpart_prob)
  
  cur_preds = np.argmax(y_pertpart_pred, axis=1)
  cur_preds = np.reshape(cur_preds,(len(y_adv_pertpart)//n_pert,n_pert),order='F')

  feed_dict = {x: x_adv_pertpart}
  y_adv_pertpart_pred = sess.run(preds,feed_dict=feed_dict)
  y_pertpart_prob_adv = sess.run(probs,feed_dict=feed_dict)
  auc_score_adv = roc_auc_score(y_test_pertpart, y_pertpart_prob_adv)
  print("auc_score_test:{},auc_score_adv:{}".format(auc_score_test, auc_score_adv))

  cur_preds_adv = np.argmax(y_adv_pertpart_pred, axis=1)
  cur_preds_adv = np.reshape(cur_preds_adv,(len(y_adv_pertpart)//n_pert,n_pert),order='F')
    
  test_result_stat=np.zeros((n_pert+1,),dtype=np.float32)
  adv_result_stat=np.zeros((n_pert+1,),dtype=np.float32)

  len_thresh = len(eva_thresh)
  distrib_incons_preds = np.zeros((len_thresh,n_pert),dtype=np.int32)
  distrib_incons_preds_adv = np.zeros((len_thresh,n_pert),dtype=np.int32)
  
  y_test5=np.argmax(y_test_pertpart,axis=1)
  y_test5=np.reshape(y_test5,(len(y_test_pertpart)//n_pert,n_pert),order='F')

  auc_div_mat = np.zeros((len(cur_preds),n_pert+1),dtype=np.int32)
  auc_div_mat_adv = np.zeros((len(cur_preds),n_pert+1),dtype=np.int32)

  for i in range(len(cur_preds)):
      temp = np.sum(np.equal(cur_preds[i,:],y_test5[i,:]))
      auc_div_mat[i,temp] = 1
      test_result_stat[temp] = test_result_stat[temp]+1
      a = np.unique(cur_preds[i,:])
      for j in range(len_thresh):
        if temp<eva_thresh[j]:
          kk = []
          for k in range(len(a)):
              kk.extend([np.sum(cur_preds[i,:]==a[k])])
          ind = np.max(np.array(kk))
          distrib_incons_preds[j,ind-1] = distrib_incons_preds[j,ind-1]+1

  y_test5=np.argmax(y_adv_pertpart,axis=1)
  y_test5=np.reshape(y_test5,(len(y_adv_pertpart)//n_pert,n_pert),order='F')
  for i in range(len(cur_preds_adv)):
      temp = np.sum(np.equal(cur_preds_adv[i,:],y_test5[i,:]))
      auc_div_mat_adv[i,temp] = 1
      adv_result_stat[temp] = adv_result_stat[temp]+1#there is a inconsensus detection results of the 27 perturbations
      a = np.unique(cur_preds_adv[i,:])
      for j in range(len_thresh):
        if temp<eva_thresh[j]:
          kk = []
          for k in range(len(a)):
              kk.extend([np.sum(cur_preds_adv[i,:]==a[k])])
          ind = np.max(np.array(kk))
          distrib_incons_preds_adv[j,ind-1] = distrib_incons_preds_adv[j,ind-1]+1

#For a benign, thare are n_pert extension images.
#And there are n_pert classifications of the extension of a benign. They may be different or same
#The maximum occurrence of the classification labels is called consistent rank.
#e.g., n_pert=5, and the classification labels of a benign are (0, 2, 2, 1, 2), then the consistent rank of the benign is 3 that is the occurrence of the label 2.
#Furthermore, correct consistent rank is the number of the extensions of a benign that are correctly classified
#test_result_stat[i]=k
#i: correct consistent rank, i=0, 1, 2, ..., n_pert-1
#k is the count of the correct consistent rank i on test images
  print("test_result_stat:{},{}".format(np.sum(test_result_stat),test_result_stat))
  print("adv_result_stat:{},{}".format(np.sum(adv_result_stat),adv_result_stat))
  for i in range(len_thresh):
#distrib_incons_preds3 is the count of consistent rank on the test images with correct consistent rank less than 3
    print("test_result cannot be classified stat (Threshold {}):{},{}".format(eva_thresh[i],np.sum(distrib_incons_preds[i,:]),distrib_incons_preds[i,:]))
#distrib_incons_preds4 is the count of consistent rank on the test images with correct consistent rank less than 4
#    print("test_result cannot be classified stat (Threshold 4):{},{}".format(np.sum(distrib_incons_preds4),distrib_incons_preds4))
#classfication: a benign with N(consistent rank)>=3 is labeled consistent rank
#The number of correctly classified benign is N(correct consistent rank)>=3
    print("The number of benigns that are correctly classified (Threshold {}):{}".format(eva_thresh[i],np.sum(test_result_stat[eva_thresh[i]-len(adv_result_stat):])))
#The number of incorrectly classified benign is the cardinality of the set {example | N(consistent rank)>=3,  true-label(example)~=consistent rank}
    print("The number of benigns that are misclassified (Threshold {}):{}".format(eva_thresh[i],np.sum(distrib_incons_preds[i,eva_thresh[i]-len(adv_result_stat):])))
    print("The number of benigns that are incorrectly detected as adv (Threshold {}):{}".format(eva_thresh[i],np.sum(distrib_incons_preds[i,:eva_thresh[i]-1])))
  
    print("adv_result cannot be classifed stat (Threshold {}):{},{}".format(eva_thresh[i],np.sum(distrib_incons_preds_adv[i,:]),distrib_incons_preds_adv[i,:]))
    print("The number of adv that are correctly classified (Threshold {}):{}".format(eva_thresh[i],np.sum(adv_result_stat[eva_thresh[i]-len(adv_result_stat):])))
    print("The number of adv that are misclassified (Threshold {}):{}".format(eva_thresh[i],np.sum(distrib_incons_preds_adv[i,eva_thresh[i]-len(adv_result_stat):])))
    print("The number of adv that are correctly detected as adv (Threshold {}):{}".format(eva_thresh[i],np.sum(distrib_incons_preds_adv[i,:eva_thresh[i]-1])))

    ####calculate auc
    benign_ind_clc = np.argwhere(np.sum(auc_div_mat[:,eva_thresh[i]:],axis=1)==1)[:,0]
    adv_ind_clc = np.argwhere(np.sum(auc_div_mat_adv[:,eva_thresh[i]:],axis=1)==1)[:,0]
    benign_inds = benign_ind_clc
    adv_inds = adv_ind_clc
    for j in range(1,n_pert):
        benign_inds = np.concatenate((benign_inds, benign_ind_clc + test_end*j), axis=0)
        adv_inds = np.concatenate((adv_inds, adv_ind_clc + test_end*j), axis=0)
    ground_labels = y_test_pertpart[tuple(benign_inds),:]
    del_ind=[]
    for j in range(nb_classes):
        if np.sum(ground_labels[:,j])==0:
            del_ind.append(j)
    del_ind = np.array(del_ind)
    ground_labels = np.delete(ground_labels,del_ind,axis=1)
    preded_probs = y_pertpart_prob[tuple(benign_inds),:]
    preded_probs = np.delete(preded_probs,del_ind,axis=1) 
    auc_score_clc_bn = roc_auc_score(ground_labels, preded_probs)
                              
    ground_labels_adv = y_adv_pertpart[tuple(adv_inds),:]
    del_ind=[]
    for j in range(nb_classes):
        if np.sum(ground_labels_adv[:,j])==0:
            del_ind.append(j)
    del_ind = np.array(del_ind)
    ground_labels_adv = np.delete(ground_labels_adv,del_ind,axis=1)
    preded_probs_adv = y_pertpart_prob_adv[tuple(adv_inds),:]
    preded_probs_adv = np.delete(preded_probs_adv,del_ind,axis=1) 
    auc_score_clc_adv = roc_auc_score(ground_labels_adv, preded_probs_adv)
#    auc_score_clc_adv = roc_auc_score(y_adv_pertpart[tuple(adv_inds),:], y_pertpart_prob_adv[tuple(adv_inds),:])
    print("(Threshold {}:) auc_score_clc_bn:{},auc_score_clc_adv:{}".format(eva_thresh[i],auc_score_clc_bn, auc_score_clc_adv))
    
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
