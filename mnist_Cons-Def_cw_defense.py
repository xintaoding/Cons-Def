"""
This tutorial shows how to implement Cons-Def against C&W white-box attacks.
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
import os
import numpy as np
import tensorflow as tf

from cleverhans.attacks import CarliniWagnerL2
from cleverhans.compat import flags
from cleverhans.dataset import MNIST
from cleverhans.utils import grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_eval, tf_model_load
from cleverhans.model_zoo.basic_cnn import ModelBasicCNN

from cleverhans.data_exten import data_exten
from sklearn.metrics import roc_curve, roc_auc_score#Added by Ding


FLAGS = flags.FLAGS
#fd = FLAGS._flags()
#kl = [keys for keys in fd]
#for keys in kl:
#      FLAGS.__delattr__(keys)

#VIZ_ENABLED = True
VIZ_ENABLED = False
BATCH_SIZE = 128
NB_EPOCHS = 6
SOURCE_SAMPLES = 10000
LEARNING_RATE = .001
CW_LEARNING_RATE = 0.01#cleverhan setting
#CW_LEARNING_RATE = .01#cw author proposed
ATTACK_ITERATIONS = 1000
MODEL_PATH = os.path.join('models', 'mnist')
#TARGETED = True
TARGETED = False

def mnist_tutorial_cw(train_start=0, train_end=60000, test_start=0,
                      test_end=10000, viz_enabled=VIZ_ENABLED,
                      nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                      source_samples=SOURCE_SAMPLES,
                      learning_rate=LEARNING_RATE,
                      attack_iterations=ATTACK_ITERATIONS,
                      model_path=MODEL_PATH,
                      targeted=TARGETED):
  """
  MNIST tutorial for Carlini and Wagner's attack
  :param train_start: index of first training set example
  :param train_end: index of last training set example
  :param test_start: index of first test set example
  :param test_end: index of last test set example
  :param viz_enabled: (boolean) activate plots of adversarial examples
  :param nb_epochs: number of epochs to train model
  :param batch_size: size of training batches
  :param nb_classes: number of output classes
  :param source_samples: number of test inputs to attack
  :param learning_rate: learning rate for training
  :param model_path: path to the model file
  :param targeted: should we run a targeted attack? or untargeted?
  """

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  # Set logging level to see debug information
  set_log_level(logging.DEBUG)

  # Create TF session
  sess = tf.Session()
  print("Created TensorFlow session.")

  set_log_level(logging.DEBUG)
  # Get MNIST data
  mnist = MNIST(train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end)
  x_train, y_train = mnist.get_set('train')
  x_test, y_test = mnist.get_set('test')
  
  # Use Image Parameters
  print("y_test_shape:{},{}".format(y_test.shape,y_test[0,:]))#########################################
  img_rows, img_cols, nchannels = x_train.shape[1:4]
  nb_classes = y_train.shape[1]
  
  print("nb_classes:{}".format(nb_classes))#########################################
#  assert 1==2

  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))
  nb_filters = 64

  # Define TF model graph
  model = ModelBasicCNN('model1', nb_classes, nb_filters)
  preds = model.get_logits(x)
#  loss = CrossEntropy(model, smoothing=0.1)
  print("Defined TensorFlow model graph.")

  print("aaaaaaaaaaaaaaaaa=+++++++++++++++++++++++++++++")
#  restname='./mnist_clean_train'
  restname='./models/mnist_models/mnist_train_2_4_8_16_32Aug50iters'
  saver = tf.train.Saver() 
  print("aaaaaaaaaaaaaaaaa=--------------------------------")
  saver.restore(sess,restname)
  print("aaaaaaaaaaaaaaaaa=************************************")
  
  eval_params = {'batch_size': batch_size}

###########################################################################
  # Craft adversarial examples using Carlini and Wagner's approach
  ###########################################################################
  nb_adv_per_sample = str(nb_classes - 1) if targeted else '1'
  print('Crafting ' + str(source_samples) + ' * ' + nb_adv_per_sample + ' adversarial examples')
  print("This could take some time ...")

  #####################################################################
  #Randomly select source_samples samples in x_test
#  test_ind=np.random.randint(0,x_test.shape[0],source_samples)
#  x_test=x_test[test_ind,:,:,:]
#  y_test=y_test[test_ind]
  print("x_test:{}",x_test.shape)
  #####################################################################
  
  # Instantiate a CW attack object
  cw = CarliniWagnerL2(model, sess=sess)
  print("cw----------------------:{}".format(cw))

  if viz_enabled:
    assert source_samples == nb_classes
    idxs = [np.where(np.argmax(y_test, axis=1) == i)[0][0]
            for i in range(nb_classes)]
    print("Iiiiiiiiiiiiiiiiiiiiiiiidxs:{}".format(idxs))
  if targeted:
    if viz_enabled:
      # Initialize our array for grid visualization
      grid_shape = (nb_classes, nb_classes, img_rows, img_cols, nchannels)
      grid_viz_data = np.zeros(grid_shape, dtype='f')
      adv_input_xs_before_target = x_test[idxs]
      adv_input_ys_before_target = y_test[idxs]
      adv_input_xs = np.array([[instance] * nb_classes for instance in x_test[idxs]], dtype=np.float32)
    else:
      adv_input_xs_before_target = x_test[:source_samples]
      adv_input_ys_before_target = y_test[:source_samples]
      adv_input_xs = np.array(
          [[instance] * nb_classes for
           instance in x_test[:source_samples]], dtype=np.float32)#produce 4-d inputs with the size of n_input*10*h*w*channel
#      print("sssssssssssssssssssssssssssssssssssssadv_input_xs:{}".format(adv_input_xs.shape))

    one_hot = np.zeros((nb_classes, nb_classes))
    one_hot[np.arange(nb_classes), np.arange(nb_classes)] = 1

    adv_input_xs = adv_input_xs.reshape((source_samples * nb_classes, img_rows, img_cols, nchannels))
    adv_input_ys = np.array([one_hot] * source_samples, dtype=np.float32).reshape((source_samples * nb_classes, nb_classes))
    yname = "y_target"
  else:
    if viz_enabled:
      # Initialize our array for grid visualization
      grid_shape = (nb_classes, 2, img_rows, img_cols, nchannels)
      grid_viz_data = np.zeros(grid_shape, dtype='f')

      adv_input_xs = x_test[idxs]
      adv_input_ys = y_test[idxs]
    else:
      adv_input_xs = x_test[:source_samples]
      adv_input_ys = y_test[:source_samples]

#    adv_input_ys = None
    yname = "y"

  if targeted:
    cw_params_batch_size = source_samples * nb_classes
  else:
    cw_params_batch_size = source_samples
  cw_params = {'binary_search_steps': 1,#'binary_search_steps': 9,#'binary_search_steps': 1,#9 is the CW author proposed, 1 is the cleverhans setting
               yname: adv_input_ys, #None
               'max_iterations': attack_iterations,
               'learning_rate': CW_LEARNING_RATE,
               'batch_size': cw_params_batch_size,
               'initial_const': 10}#'initial_const': 10#10 is the default set of cleverhans,0.001 is the CW author proposed

  print("aaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbax_test:{}".format(x_test.shape))
  adv = cw.generate_np(adv_input_xs, **cw_params)
  
  if restname=='mnist_clean_train':
      np.save("mnist_cwadv_x_test",adv)
      np.save("mnist_cwadv_y_test",y_test)
      np.save("mnist_cw_x_test",adv_input_xs)
      print("adversarial examples produced on clean model are saved. adv:{},y_test:{}".format(adv.shape,y_test.shape))

  #  print("adv----------------------:{},targeted:{},adv:{}".format(adv.shape,targeted,adv.shape))

  eval_params = {'batch_size': np.minimum(nb_classes, source_samples)}
#  accuracy,suc_att_exam = model_eval(sess, x, y, preds, adv_input_xs, adv_input_ys, args=eval_params)
  adv_acc,adv_suc_att_exam = model_eval(sess, x, y, preds, adv, adv_input_ys, args=eval_params)
  #adv_input_ys is the ground label, for target attack, it is fixed as [[1,0..,0],[0,1,0..,0],...]
 #adv_accuracy is the sucessful attack ratio of the number of successful attacks to the number of total attacks
  if targeted:
      accuracy,suc_att_exam = model_eval(sess, x, y, preds, adv_input_xs_before_target, adv_input_ys_before_target, args=eval_params)
      adv_accuracy = adv_acc
  else:
      accuracy,suc_att_exam = model_eval(sess, x, y, preds, adv_input_xs, adv_input_ys, args=eval_params)
      adv_accuracy = 1 - adv_acc 

  if viz_enabled:
    for j in range(nb_classes):
      if targeted:
        for i in range(nb_classes):
          grid_viz_data[i, j] = adv[i * nb_classes + j]
      else:
        grid_viz_data[j, 0] = adv_input_xs[j]
        grid_viz_data[j, 1] = adv[j]

    print(grid_viz_data.shape)
  print('--------------------------------------')

  assert x_test.shape[0] == test_end - test_start, x_test.shape
  print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
#  report.clean_train_clean_eval = accuracy

  # Compute the number of adversarial examples that were successfully found
#  report.clean_train_adv_eval = 1. - adv_accuracy
  print('Test accuracy on adversarial examples: %0.4f' % (adv_acc))
  print('Avg. rate of successful adv. examples /ASR {0:.4f}'.format(adv_accuracy))

  # Compute the average distortion introduced by the algorithm
  percent_perturbed = np.mean(np.sum((adv - adv_input_xs)**2, axis=(1, 2, 3))**.5)

  #for untargeted attack, suc_att_exam[i] is true means a successful classified examples
  #for targeted attack, suc_att_exam[i] is true means a successful attack, it counts succeful attacked examples
  dsae=0
  kk = 0
  if targeted:
      for i in range(len(adv_suc_att_exam)):
          if adv_suc_att_exam[i]>0:
              dsae+=np.sum((adv[i,:,:,:] - adv_input_xs[i,:,:,:])**2)**.5
      dsae=dsae/sum(adv_suc_att_exam)
      print("For targeted attack, the number of misclassified examples: sum(adv_suc_att_exam):{}".format(sum(adv_suc_att_exam)))
  else:
      for i in range(len(adv_suc_att_exam)):
          if adv_suc_att_exam[i]==0 and suc_att_exam[i]>0:
              dsae+=np.sum((adv[i,:,:,:] - adv_input_xs[i,:,:,:])**2)**.5
              kk += 1
      dsae=dsae/(kk+1.e-20)
      print("For untargeted attack, the number of misclassified examples: sum(adv_suc_att_exam==0):{}".format(np.sum(adv_suc_att_exam==0)))
                              
  

  print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))
  print('The number of successful attack:{}, Avg. L_2 norm of perturbations on successful attack / dsae:{}'.format(kk,dsae))

###########################################################################
#  adv = np.round(adv*256)/256.0
  pad_size=3
  adv34 = np.pad(adv,((pad_size,pad_size),(pad_size,pad_size),(0,0)),'reflect')
  test34 = np.pad(adv_input_xs,((pad_size,pad_size),(pad_size,pad_size),(0,0)),'reflect')
  for i in range(len(adv)):
      tf_image = adv34[i,:,:,:]
      lu1 = np.random.randint(0,6)
      lu2 = np.random.randint(0,6)
      adv[i,:,:,:] = tf_image[lu1:lu1+img_rows,lu2:lu2+img_cols,:]
      adv_input_xs[i,:,:,:]=test34[i,lu1:lu1+img_rows,lu2:lu2+img_cols,:]

  n_pert = 5#5 aug  test
  eva_thresh = np.linspace(3,5,3).astype('int32')#5 aug test
###########################################################################    
 
  n_exam = len(adv) 
  x_adv_extended, y_adv_extended = data_exten(adv, adv_input_ys, n_exam, nb_classes, img_rows, img_cols, 1)#
  x_adv_pertpart = x_adv_extended[:n_pert*n_exam,:,:,:]
  y_adv_pertpart = y_adv_extended[:n_pert*n_exam,:]
  if targeted:
      n_exam = len(adv_input_ys_before_target)
      x_test_extended, y_test_extended = data_exten(adv_input_xs_before_target, adv_input_ys_before_target, n_exam, nb_classes, img_rows, img_cols, 1)
  else:
      n_exam = len(adv_input_ys)
      x_test_extended, y_test_extended = data_exten(adv_input_xs, adv_input_ys, n_exam, nb_classes, img_rows, img_cols, 1)
  x_test_pertpart = x_test_extended[:n_pert*n_exam,:,:,:]
  y_test_pertpart = y_test_extended[:n_pert*n_exam,:]

  acc_ext,suc_att_ext = model_eval(sess, x, y, preds, x_test_pertpart, y_test_pertpart, args=eval_params)
  print('Test accuracy on legitimate examples extened by x_test: %0.4f' % (acc_ext))

  acc_adv_ext,suc_att_adv_ext = model_eval(sess, x, y, preds, x_adv_pertpart, y_adv_pertpart, args=eval_params)
  print('Test accuracy on extended examples of adversarials: %0.4f' % (acc_adv_ext))

  feed_dict = {x: x_test_pertpart}
  y_pertpart_pred = sess.run(preds,feed_dict=feed_dict)
  probs = model.get_probs(x)
  y_pertpart_prob = sess.run(probs,feed_dict=feed_dict)
  auc_score_test = roc_auc_score(y_test_pertpart, y_pertpart_prob)
  
  cur_preds = np.argmax(y_pertpart_pred, axis=1)
  cur_preds = np.reshape(cur_preds,(len(y_test_pertpart)//n_pert,n_pert),order='F')

  feed_dict = {x: x_adv_pertpart}
  y_adv_pertpart_pred = sess.run(preds,feed_dict=feed_dict)
  y_pertpart_prob_adv = sess.run(probs,feed_dict=feed_dict)
  auc_score_adv = roc_auc_score(y_adv_pertpart, y_pertpart_prob_adv)
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
        benign_inds = np.concatenate((benign_inds, benign_ind_clc + source_samples*j), axis=0)
        adv_inds = np.concatenate((adv_inds, adv_ind_clc + source_samples*j), axis=0)
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
                              
    ground_labels = y_adv_pertpart[tuple(adv_inds),:]
    del_ind=[]
    for j in range(nb_classes):
        if np.sum(ground_labels[:,j])==0:
            del_ind.append(j)
    del_ind = np.array(del_ind)
    ground_labels = np.delete(ground_labels,del_ind,axis=1)
    preded_probs = y_pertpart_prob_adv[tuple(adv_inds),:]
    preded_probs = np.delete(preded_probs,del_ind,axis=1) 
    auc_score_clc_adv = roc_auc_score(ground_labels, preded_probs)
#    auc_score_clc_adv = roc_auc_score(y_adv_pertpart[tuple(adv_inds),:], y_pertpart_prob_adv[tuple(adv_inds),:])
    print("(Threshold {}:) auc_score_clc_bn:{},auc_score_clc_adv:{}".format(eva_thresh[i],auc_score_clc_bn, auc_score_clc_adv))

    

  # Close TF session
  sess.close()
  
  # Finally, block & display a grid of all the adversarial examples
  if viz_enabled:
    _ = grid_visual(grid_viz_data)

def main(argv=None):
  from cleverhans_tutorials import check_installation
  check_installation(__file__)

  mnist_tutorial_cw(viz_enabled=FLAGS.viz_enabled,
                    nb_epochs=FLAGS.nb_epochs,
                    batch_size=FLAGS.batch_size,
                    source_samples=FLAGS.source_samples,
                    learning_rate=FLAGS.learning_rate,
                    attack_iterations=FLAGS.attack_iterations,
                    model_path=FLAGS.model_path,
                    targeted=FLAGS.targeted)


if __name__ == '__main__':
  
  flags.DEFINE_boolean('viz_enabled', VIZ_ENABLED,
                       'Visualize adversarial ex.')
  flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                       'Number of epochs to train model')
  flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Size of training batches')
  flags.DEFINE_integer('source_samples', SOURCE_SAMPLES,
                       'Number of test inputs to attack')
  flags.DEFINE_float('learning_rate', LEARNING_RATE,
                     'Learning rate for training')
  flags.DEFINE_string('model_path', MODEL_PATH,
                      'Path to save or load the model file')
  flags.DEFINE_integer('attack_iterations', ATTACK_ITERATIONS,
                       'Number of iterations to run attack; 1000 is good')
  flags.DEFINE_boolean('targeted', TARGETED,
                       'Run the tutorial in targeted mode?')

  tf.app.run()
